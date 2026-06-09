import os
import monai

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers import LlamaTokenizer
from torchvision.ops import sigmoid_focal_loss
import torch.distributed as dist

from Customization import (compute_outputs_from_model, get_customized_loss, other_model_operations)
from ReferDataset import get_data_loader
from Model import Model
from Utils import (compute_IoU_2class_batch, compute_IoU_1class_batch, compute_dice_from_iou, log_message, set_logger, format_string)
from Config import TokenizerPath


class Solver():
    def __init__(self, opt):
        self.opt = opt
        
        self.task_path, self.writer, self.best_valid_model_path = self.prepare_checkpoint_log()
        log_message("Making logger...", self.opt.local_rank)
        log_message(str(self.opt), self.opt.local_rank)
        
        log_message("Making model...", self.opt.local_rank)
        tokenizer = LlamaTokenizer.from_pretrained(TokenizerPath, add_bos_token=True, add_eos_token=False)
        tokenizer.padding_side = opt.tokenizer_padding_side
        if opt.tokenizer_pad_key == 'unk':
            tokenizer.pad_token = tokenizer.unk_token
        elif opt.tokenizer_pad_key == 'eos':
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise NotImplementedError
        self.opt.tokenizer = tokenizer
        self.model = Model(self.opt, tokenizer).to(opt.device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        other_model_operations(self.model, self.opt)
        self.model = DDP(self.model, device_ids=[opt.gpu_id], find_unused_parameters=False)
        
        log_message("Making dataset...", self.opt.local_rank)
        self.train_loader, self.inference_loaders = get_data_loader(self.opt)
        
        log_message("Making optimizer...", self.opt.local_rank)
        self.optimizer, self.lr_schedule = self.get_optimizer(self.model)
        self.loss_functions = [self.get_task_loss(), get_customized_loss(self.opt)]

    def solve(self):
        log_message("Start training...", self.opt.local_rank)
        best_scores, best_valid_model_states = dict(), dict()
        for val_subset in list(self.inference_loaders.keys()):
            best_scores[val_subset], best_valid_model_states[val_subset] = None, None

        for epoch in range(self.opt.epochs_num):
            self.train_loader.sampler.set_epoch(epoch)
            for val_subset in list(self.inference_loaders.keys()):
                self.inference_loaders[val_subset].sampler.set_epoch(epoch)
                
            train_loss, train_score = self.train(self.train_loader) # val_losses and val_scores are not dict, just objects
            val_losses, val_scores = self.evaluate(self.inference_loaders) # val_losses and val_scores are dict, keys are val_subsets
                        
            if self.opt.warmup_ratio <= 0 and self.opt.lr_decrease != 'seg': # If using warming up, we use batch-based lr scheduler
                self.lr_schedule.step()

            for val_subset in list(self.inference_loaders.keys()):
                if self.current_result_better(best_scores[val_subset], val_scores[val_subset]):
                    log_message('Better '+ val_subset+' score found...', self.opt.local_rank)
                    best_scores[val_subset] = val_scores[val_subset]
                    best_valid_model_states[val_subset] = {"epoch": epoch, "model": self.model.state_dict(), "optim": self.optimizer.state_dict()}
            
            torch.distributed.barrier()
            epoch_summary = self.build_message(epoch, train_loss, train_score, val_losses, val_scores)
            log_message(epoch_summary, self.opt.local_rank)
            self.log_tf_board(epoch, train_loss, train_score, val_losses, val_scores)

        # Saving results
        log_message("Training complete.", self.opt.local_rank)
        if self.writer is not None:
            self.writer.close()
        self.log_best_scores(best_scores)
        self.save_results(best_valid_model_states)

    def prepare_checkpoint_log(self):
        if self.opt.local_rank !=0:
            return None, None, None
        task_path = os.path.join('./Running/', self.opt.task_name)
        best_valid_model_path = os.path.join(task_path, "best_model.pth.tar")

        os.makedirs(task_path, exist_ok=True)
        set_logger(os.path.join(task_path, "Running.log"))
        
        import glob
        previous_writer_files = glob.glob(os.path.join(task_path, 'events.out.tfevents*'))
        for f in previous_writer_files:
            os.remove(f)

        writer = SummaryWriter(task_path)
        return task_path, writer, best_valid_model_path
    
    def get_optimizer(self, model):

        def get_params(model):
            prompt_results, decoder_results = [], []
            for p in model.named_parameters():
                if p[1].requires_grad and 'decoder' in p[0]:
                    decoder_results.append(p[1])
                elif p[1].requires_grad and 'decoder' not in p[0]:
                    prompt_results.append(p[1])
                else:
                    assert p[1].requires_grad == False
            return prompt_results, decoder_results
        

        def get_all_params(model):
            results = []
            for p in model.named_parameters():
                if p[1].requires_grad:
                    results.append(p[1])
            return results
        
        prompt_results, decoder_results = get_params(model)
        params = [
            {'params': prompt_results, 'lr': self.opt.learning_rate, },
            {'params': decoder_results, 'lr': self.opt.learning_rate*self.opt.decoder_lr_decay},
        ]
        
        if self.opt.optm == "Adam":
            optimizer = torch.optim.Adam(params, lr=float(self.opt.learning_rate), weight_decay=self.opt.weight_decay,)# eps=1e-4)
        elif self.opt.optm == "SGD":
            optimizer = torch.optim.SGD(params, lr=float(self.opt.learning_rate), weight_decay=self.opt.weight_decay, momentum=0.9)
        elif self.opt.optm == "AdamW":
            optimizer = torch.optim.AdamW(params, lr=float(self.opt.learning_rate), weight_decay=self.opt.weight_decay, amsgrad=self.opt.amsgrad, )#eps=1e-4)
        else:
            raise NotImplementedError

        if self.opt.warmup_ratio > 0:
            lr_schedule = get_cosine_schedule_with_warmup(optimizer, 
                        num_warmup_steps=int(self.opt.warmup_ratio*self.opt.epochs_num*len(self.train_loader)), num_training_steps=self.opt.epochs_num*len(self.train_loader))
        else:
            if self.opt.lr_decrease == 'step':
                self.opt.lr_decrease_iter = int(self.opt.lr_decrease_iter)
                lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, self.opt.lr_decrease_iter, self.opt.lr_decrease_rate)
            elif self.opt.lr_decrease == 'multi_step':
                self.opt.lr_decrease_iter = list((map(int, self.opt.lr_decrease_iter.split('-'))))
                lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.opt.lr_decrease_iter, self.opt.lr_decrease_rate)
            elif self.opt.lr_decrease == 'exp':
                lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.opt.lr_decrease_rate)
            elif self.opt.lr_decrease == 'seg':
                lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(self.train_loader) * self.opt.epochs_num)) ** 0.9)
            else:
                raise NotImplementedError
        return optimizer, lr_schedule

    def get_task_loss(self):
        if self.opt.loss == 'CE':
            weight = torch.FloatTensor([0.9, 1.1]).to(self.opt.device)
            loss_func = torch.nn.CrossEntropyLoss(weight=weight)
        elif self.opt.loss == 'BCE':
            # weight = torch.FloatTensor([0.9, 1.1]).to(self.opt.device)
            loss_func = torch.nn.BCEWithLogitsLoss()
        elif self.opt.loss == 'DICE':
            loss_func = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        elif self.opt.loss == 'Focal':
            loss_func = sigmoid_focal_loss
        else :
            raise NotImplementedError
            
        return loss_func

    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0
        all_ious, all_interactions, all_unions = 0, 0, 0
        precision_iou_threshold, precision_iou_correct, precision_iou = [0.5, 0.6, 0.7, 0.8, 0.9], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        all_dices = 0
        
        num_samples = 0 # We cannot directly get num_samples by len(dataset), because of DDP

        for i, datas in enumerate(train_loader):
            images, labels, sentences = datas
            images, labels = images.to(self.opt.device), labels.to(self.opt.device)
            outputs = compute_outputs_from_model(self.model, images, sentences, self.opt)
            loss = self.compute_loss(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            if self.opt.check_gradient:
                self.check_gradient()
            if self.opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.opt.gradient_clip)
            if self.opt.gradient_norm > 0:
                torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad], self.opt.gradient_norm)
            self.optimizer.step()
            if self.opt.warmup_ratio > 0 or self.opt.lr_decrease == 'seg':
                self.lr_schedule.step()
            
            running_loss += loss.item()
            predictions = outputs[0]
            if self.opt.num_class == 2:
                ious_batch, interactions_batch, unions_batch = compute_IoU_2class_batch(predictions, labels)
            elif self.opt.num_class == 1:
                ious_batch, interactions_batch, unions_batch = compute_IoU_1class_batch(predictions, labels)
            else:
                raise NotImplementedError
            all_ious, all_interactions, all_unions = all_ious+ious_batch.sum(), all_interactions+interactions_batch.sum(), all_unions+unions_batch.sum()
            all_dices = all_dices + compute_dice_from_iou(ious_batch).sum()
            
            # These codes are borrowed from others. I don't thinks the origianl codes are correct, because it calculate the precision on batch-level.
            # So I have modified them.
            for n_threshold in range(len(precision_iou_threshold)):
                precision_iou_correct[n_threshold] += (ious_batch >= precision_iou_threshold[n_threshold]).long().sum()
                
            num_samples += labels.shape[0]
            
            if i % 100 == 0:
                log_message('Iter-{}[{}/{}]: loss[{:5.2f}] Mask[{:5.2f}] V[{:5.2f}] T[{:5.2f}] V0[{:5.2f}] Ekm[{:5.2f}] Ekp[{:5.2f}] OIoU[{:5.2f}]'.format(
                    'Train', i, len(train_loader), loss.item(), predictions.sum().item(), outputs[1].sum().item(), outputs[2].sum().item(), 
                    outputs[3].sum().item(), outputs[4].sum().item(), outputs[5].sum().item(), (all_interactions / all_unions).item()), self.opt.local_rank)
            
            if self.opt.check_gradient:
                exit(0)
        
        running_loss = running_loss / len(train_loader)
        
        # torch.tensor is used to cast value to tensors, torch.Tensor recieves shapes and random generate some.
        running_loss     = torch.tensor(running_loss).to(self.opt.device)
        all_ious         = all_ious.to(self.opt.device)
        all_interactions = all_interactions.to(self.opt.device)
        all_unions       = all_unions.to(self.opt.device)
        all_dices        = all_dices.to(self.opt.device)
        num_samples      = torch.tensor(num_samples).to(self.opt.device)
        precision_iou_correct = torch.tensor(precision_iou_correct).to(self.opt.device)
        self.reduce_metrics([running_loss], [all_ious, all_interactions, all_unions, all_dices, num_samples, precision_iou_correct])
        
        all_ious_mean = (all_ious / num_samples).item()
        overall_iou = (all_interactions / all_unions).item()
        for n_precision in range(len(precision_iou)):
            precision_iou[n_precision] = precision_iou_correct[n_precision] / num_samples
        all_dices_mean = (all_dices / num_samples).item()
        
        train_score = self.get_score_from_result(all_ious_mean, overall_iou, precision_iou, precision_iou_threshold, all_dices_mean)

        return running_loss, train_score

    def evaluate(self, inference_loaders):
        self.model.eval()
        all_val_losses, all_val_scores = dict(), dict()
        
        with torch.no_grad():
            for val_subset in list(inference_loaders.keys()):
                subset_loader = inference_loaders[val_subset]
                running_loss = 0.0
                all_ious, all_interactions, all_unions = 0, 0, 0
                precision_iou_threshold, precision_iou_correct, precision_iou = [0.5, 0.6, 0.7, 0.8, 0.9], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
                all_dices = 0
                
                num_samples = 0 # We cannot directly get num_samples by len(dataset), because of DDP

                for i, datas in enumerate(subset_loader):
                    images, labels, sentences = datas
                    images, labels = images.to(self.opt.device), labels.to(self.opt.device)
                    outputs = compute_outputs_from_model(self.model, images, sentences, self.opt)
                    loss = self.compute_loss(outputs, labels)
                                        
                    running_loss += loss.item()
                    predictions = outputs[0]
                    if self.opt.num_class == 2:
                        ious_batch, interactions_batch, unions_batch = compute_IoU_2class_batch(predictions, labels)
                    elif self.opt.num_class == 1:
                        ious_batch, interactions_batch, unions_batch = compute_IoU_1class_batch(predictions, labels)
                    else:
                        raise NotImplementedError
                    all_ious, all_interactions, all_unions = all_ious+ious_batch.sum(), all_interactions+interactions_batch.sum(), all_unions+unions_batch.sum()
                    all_dices = all_dices + compute_dice_from_iou(ious_batch).sum()
                    
                    # These codes are borrowed from others. I don't thinks the origianl codes are correct, because it calculate the precision on batch-level.
                    # So I have modified them.
                    for n_threshold in range(len(precision_iou_threshold)):
                        precision_iou_correct[n_threshold] += (ious_batch >= precision_iou_threshold[n_threshold]).long().sum()
                        
                    num_samples += labels.shape[0]
                    
                    if i % 100 == 0:
                        log_message('Iter-{}[{}/{}]: loss[{:5.2f}]'.format(val_subset, i, len(subset_loader),loss.item()), self.opt.local_rank)
                
                running_loss = running_loss / len(subset_loader)
                
                # torch.tensor is used to cast value to tensors, torch.Tensor recieves shapes and random generate some.
                running_loss     = torch.tensor(running_loss).to(self.opt.device)
                all_ious         = all_ious.to(self.opt.device)
                all_interactions = all_interactions.to(self.opt.device)
                all_unions       = all_unions.to(self.opt.device)
                all_dices        = all_dices.to(self.opt.device)
                num_samples      = torch.tensor(num_samples).to(self.opt.device)
                precision_iou_correct = torch.tensor(precision_iou_correct).to(self.opt.device)
                self.reduce_metrics([running_loss], [all_ious, all_interactions, all_unions, all_dices, num_samples, precision_iou_correct])

                all_ious_mean = (all_ious / num_samples).item()
                overall_iou = (all_interactions / all_unions).item()
                for n_precision in range(len(precision_iou)):
                    precision_iou[n_precision] = precision_iou_correct[n_precision] / num_samples
                all_dices_mean = (all_dices / num_samples).item()
                                
                subset_score = self.get_score_from_result(all_ious_mean, overall_iou, precision_iou, precision_iou_threshold, all_dices_mean)
                
                all_val_scores[val_subset] = subset_score
                all_val_losses[val_subset] = running_loss

        return all_val_losses, all_val_scores

    def reduce_metrics(self, avg_metrics, sum_metrics):
        for metric in avg_metrics:
            dist.all_reduce(metric, dist.ReduceOp.AVG)
        for metric in sum_metrics:
            dist.all_reduce(metric, dist.ReduceOp.SUM)

    def compute_loss(self, outputs, labels):
        predictions = outputs[0]
        task_loss_function = self.loss_functions[0]
        
        if self.opt.loss in ['CE']: # predictions: [bs, num_class, h, w], labels: [bs, h, w]
            assert predictions.shape[1] == self.opt.num_class
            task_loss = task_loss_function(predictions, labels)
        elif self.opt.loss in ['BCE']: # predictions: [bs, h, w], labels: [bs, h, w]
            predictions = predictions.squeeze(1)
            assert len(predictions.shape) == 3 
            # print(predictions.shape, labels)
            task_loss = task_loss_function(predictions, labels.float())
        elif self.opt.loss in ['DICE']: # predictions: [bs, 1, h, w], labels: [bs, 1, h, w]
            labels = labels.unsqueeze(1)
            assert predictions.shape[1] == 1 and labels.shape[1] == 1 
            task_loss = task_loss_function(predictions, labels)
        elif self.opt.loss in ['Focal']: # predictions: [bs, 1, h, w], labels: [bs, 1, h, w]
            labels = labels.unsqueeze(1)
            assert predictions.shape[1] == 1 and labels.shape[1] == 1 
            task_loss = task_loss_function(predictions, labels.float(), reduction='mean')
        else:
            raise NotImplementedError

        all_loss = task_loss
        return all_loss

    def check_gradient(self):
        for name, parms in self.model.named_parameters():
            if parms.requires_grad:
                msg = 'Name/GradRequire/Param/Grad: {}/{}/{:5.2f}/{:5.2f}'.format(name, parms.requires_grad, parms.sum(), parms.grad.sum() if parms.requires_grad else 0)
                log_message(msg, self.opt.local_rank)

    def get_score_from_result(self, all_ious_mean, overall_iou, precision_iou, precision_iou_threshold, dice_mean):
        score = {
            'MIoU': all_ious_mean,
            'OIoU': overall_iou,
            'MDICE': dice_mean,
        }
        for threshold, precision in zip(precision_iou_threshold, precision_iou):
            score['Pcs@{:.1f}'.format(threshold)] = precision
        return score
        
    def current_result_better(self, best_score, current_score):
        if best_score is None:
            return True
        else:
            return current_score['OIoU'] > best_score['OIoU']

    def build_message(self, epoch, train_loss, train_score, val_losses, val_scores):
        msg = "Epoch:[{:3.0f}]\n".format(epoch + 1)       
        
        msg += "Train:"
        msg += " Loss:[{0:.3f}]".format(train_loss)
        msg += " MIoU/OIoU/MDICE:[{0:6.3f}/{1:6.3f}/{2:6.3f}]".format(train_score['MIoU'], train_score['OIoU'], train_score['MDICE'])
        msg += " Pcs@0.5/0.6/0.7/0.8/0.9:[{0:6.3f}/{1:6.3f}/{2:6.3f}/{3:6.3f}/{4:6.3f}]".format(train_score['Pcs@0.5'], train_score['Pcs@0.6'], train_score['Pcs@0.7'], train_score['Pcs@0.8'], train_score['Pcs@0.9'])
        msg += '\n'

        for val_subset in list(val_losses.keys()):
            msg += "{}:".format(format_string(val_subset))
            msg += " Loss:[{0:.3f}]".format(val_losses[val_subset])
            msg += " MIoU/OIoU/MDICE:[{0:6.3f}/{1:6.3f}/{2:6.3f}]".format(val_scores[val_subset]['MIoU'], val_scores[val_subset]['OIoU'], val_scores[val_subset]['MDICE'])
            msg += " Pcs@0.5/0.6/0.7/0.8/0.9:[{0:6.3f}/{1:6.3f}/{2:6.3f}/{3:6.3f}/{4:6.3f}]".format(val_scores[val_subset]['Pcs@0.5'], val_scores[val_subset]['Pcs@0.6'], val_scores[val_subset]['Pcs@0.7'], val_scores[val_subset]['Pcs@0.8'], val_scores[val_subset]['Pcs@0.9'])
            msg += '\n'

        return msg

    def build_single_message(self, best_score, mode):
        msg = mode
        for key in best_score.keys():
            msg += " "+key+":[{0:6.3f}]".format(best_score[key])
        return msg

    def log_tf_board(self, epoch, train_loss, train_score, val_losses, val_scores):
        if self.opt.local_rank != 0:
            return
        self.writer.add_scalar('Train/Loss', train_loss, epoch)
        for key in train_score.keys():
            self.writer.add_scalar('Train/'+key, train_score[key], epoch)
        
        for val_subset in list(val_losses.keys()):
            self.writer.add_scalar(format_string(val_subset)+'/Loss', val_losses[val_subset], epoch)
            for key in val_scores[val_subset].keys():
                self.writer.add_scalar(format_string(val_subset)+'/'+key, val_scores[val_subset][key], epoch)
        
        self.writer.add_scalar('Lr',  self.lr_schedule.get_last_lr()[-1], epoch)

    def log_best_scores(self, best_scores):
        for val_subset in list(best_scores.keys()):
            log_message(self.build_single_message(best_scores[val_subset], 'Best '+format_string(val_subset)+' Score: \t'), self.opt.local_rank)

    def save_results(self, best_valid_model_states):
        if self.opt.local_rank != 0:
            return
        torch.save(best_valid_model_states, self.best_valid_model_path)
