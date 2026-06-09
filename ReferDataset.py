import os
import torch
import numpy as np
from PIL import Image
from refer import REFER
import transforms
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPProcessor
from Config import ModelCLIPPATH
from Config import modes
# Usage:
# For dataset=refcoco and refcoco+, splitBy in [unc], split in [train, val, testA, and testB]
# For dataset=refcocog, splitBy in [google], split in [train, val]
# For dataset=refcocog, splitBy in [umd], split in [train, val, test]
# For dataset=refclef, splitBy in [unc], split in [train, val, test]

# MUST use the two parameters: use_square_size=True, do_center_crop=False
processor = CLIPProcessor.from_pretrained(ModelCLIPPATH, use_square_size=True, do_center_crop=False)


class ReferDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, splitBy='unc', image_transforms=None, target_transforms=None, split='train', eval_mode=False):
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        assert split in modes
        self.refer = REFER(dataset, splitBy)

        self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.eval_mode = eval_mode # if we are testing on a dataset, test all sentences of an object; o/w, we are validating during training, randomly sample one sentence for efficiency
        self.raw_sentences = []
        for r in ref_ids:
            ref = self.refer.Refs[r]
            
            raw_sentences_each = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                raw_sentences_each.append(el['raw'])

            self.raw_sentences.append(raw_sentences_each)

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name']))
        img_rgb = img.convert("RGB")
        img_my = processor(images=img, return_tensors="pt",)['pixel_values'].squeeze()

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(img_rgb, annot)

        sentences = []
        if self.eval_mode:
            for s in self.raw_sentences[index]:
                sentences.append(s)
        else:
            choice_sent = np.random.choice(len(self.raw_sentences[index]))
            # sentences = self.raw_sentences[index][choice_sent]
            sentences.append(self.raw_sentences[index][choice_sent])

        return img_my, target, sentences


class ReferDatasetAll(torch.utils.data.Dataset):
    def __init__(self, dataset, splitBy='unc', image_transforms=None, target_transforms=None, split='train', eval_mode=None):
        # eval_mode is useless 
        
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(dataset, splitBy)

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids
            
        self.raw_sentences = []
        self.raw_sentences_original_indice = []
        counter = 0
        for r in ref_ids:
            ref = self.refer.Refs[r]
            
            for sentence in ref['sentences']:
                self.raw_sentences.append(sentence['raw'])
                self.raw_sentences_original_indice.append(counter)
            
            counter += 1
        assert len(self.raw_sentences) == len(self.raw_sentences_original_indice)
        
    def __len__(self):
        return len(self.raw_sentences)

    def __getitem__(self, new_index):
        original_index = self.raw_sentences_original_indice[new_index]
        
        this_ref_id = self.ref_ids[original_index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name']))
        img_rgb = img.convert("RGB")
        img_my = processor(images=img, return_tensors="pt",)['pixel_values'].squeeze()

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            img, target = self.image_transforms(img_rgb, annot)

        # print('inside', img.shape, img_my.shape)
        sentence = self.raw_sentences[new_index]

        return img_my, target, [sentence]


def multi_collate_refseg(batch):
    # batch is a list with length: batch_size, each element is a list containing the data returned in dataset.
    images = [data[0] for data in batch]
    target = [data[1] for data in batch]
    sentences = [data[2] for data in batch]
    
    images = torch.stack(images, dim=0)
    target = torch.stack(target, dim=0)
        
    return images, target, sentences


def get_data_loader(opt):
    transforms_for_image = transforms.Compose([
        transforms.Resize(opt.input_size, opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    train_dataset_class = ReferDatasetAll if opt.train_with_all else ReferDataset
    valid_dataset_class = ReferDatasetAll if opt.test_with_all else ReferDataset
    
    if opt.dataset in ['refcoco', 'refcoco+']:
        dataset_train = train_dataset_class(dataset=opt.dataset, splitBy='unc', image_transforms=transforms_for_image, target_transforms=None, split='train', eval_mode=False)
        dataset_val =   valid_dataset_class(dataset=opt.dataset, splitBy='unc', image_transforms=transforms_for_image, target_transforms=None, split='val',   eval_mode=False)
        dataset_testA = valid_dataset_class(dataset=opt.dataset, splitBy='unc', image_transforms=transforms_for_image, target_transforms=None, split='testA', eval_mode=False)
        dataset_testB = valid_dataset_class(dataset=opt.dataset, splitBy='unc', image_transforms=transforms_for_image, target_transforms=None, split='testB', eval_mode=False)
        sampler_train = DistributedSampler(dataset_train, shuffle=True,  drop_last=opt.drop_last)
        sampler_val   = DistributedSampler(dataset_val,   shuffle=False, drop_last=opt.drop_last)
        sampler_testA = DistributedSampler(dataset_testA, shuffle=False, drop_last=opt.drop_last)
        sampler_testB = DistributedSampler(dataset_testB, shuffle=False, drop_last=opt.drop_last)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, collate_fn=multi_collate_refseg, drop_last=opt.drop_last, sampler=sampler_train,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)
        data_loader_val =   torch.utils.data.DataLoader(dataset_val,   batch_size=opt.batch_size, collate_fn=multi_collate_refseg, drop_last=opt.drop_last, sampler=sampler_val,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)
        data_loader_testA = torch.utils.data.DataLoader(dataset_testA, batch_size=opt.batch_size, collate_fn=multi_collate_refseg, drop_last=opt.drop_last, sampler=sampler_testA,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)
        data_loader_testB = torch.utils.data.DataLoader(dataset_testB, batch_size=opt.batch_size, collate_fn=multi_collate_refseg, drop_last=opt.drop_last, sampler=sampler_testB,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)

        dataloaders_inference = {
            'val': data_loader_val,
            'testA': data_loader_testA,
            'testB': data_loader_testB,
        }
        return data_loader_train, dataloaders_inference
    elif opt.dataset in ['refcocog'] and opt.splitBy in ['google']:
        dataset_train = train_dataset_class(dataset=opt.dataset, splitBy='google', image_transforms=transforms_for_image, target_transforms=None, split='train', eval_mode=False)
        dataset_val =   valid_dataset_class(dataset=opt.dataset, splitBy='google', image_transforms=transforms_for_image, target_transforms=None, split='val',   eval_mode=False)
        sampler_train = DistributedSampler(dataset_train, shuffle=True,  drop_last=opt.drop_last)
        sampler_val   = DistributedSampler(dataset_val,   shuffle=False, drop_last=opt.drop_last)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, collate_fn=multi_collate_refseg, drop_last=opt.drop_last, sampler=sampler_train,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)
        data_loader_val =   torch.utils.data.DataLoader(dataset_val,   batch_size=opt.batch_size, collate_fn=multi_collate_refseg,drop_last=opt.drop_last, sampler=sampler_val,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)
        dataloaders_inference = {
            'val': data_loader_val,
        }
        return data_loader_train, dataloaders_inference
    elif opt.dataset in ['refcocog'] and opt.splitBy in ['umd']:
        dataset_train = train_dataset_class(dataset=opt.dataset, splitBy='umd', image_transforms=transforms_for_image, target_transforms=None, split='train', eval_mode=False)
        dataset_val =   valid_dataset_class(dataset=opt.dataset, splitBy='umd', image_transforms=transforms_for_image, target_transforms=None, split='val',   eval_mode=False)
        dataset_test  = valid_dataset_class(dataset=opt.dataset, splitBy='umd', image_transforms=transforms_for_image, target_transforms=None, split='test',  eval_mode=False)
        sampler_train = DistributedSampler(dataset_train, shuffle=True,  drop_last=opt.drop_last)
        sampler_val   = DistributedSampler(dataset_val,   shuffle=False, drop_last=opt.drop_last)
        sampler_test  = DistributedSampler(dataset_test,  shuffle=False, drop_last=opt.drop_last)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, collate_fn=multi_collate_refseg, drop_last=opt.drop_last, sampler=sampler_train,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)
        data_loader_val =   torch.utils.data.DataLoader(dataset_val,   batch_size=opt.batch_size, collate_fn=multi_collate_refseg, drop_last=opt.drop_last, sampler=sampler_val,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)
        data_loader_test =  torch.utils.data.DataLoader(dataset_test,  batch_size=opt.batch_size, collate_fn=multi_collate_refseg, drop_last=opt.drop_last, sampler=sampler_test,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)
        dataloaders_inference = {
            'val': data_loader_val,
            'test': data_loader_test,
        }
        return data_loader_train, dataloaders_inference
    elif opt.dataset in ['refclef']:
        dataset_train = train_dataset_class(dataset=opt.dataset, splitBy='unc', image_transforms=transforms_for_image, target_transforms=None, split='train', eval_mode=False)
        dataset_val =   valid_dataset_class(dataset=opt.dataset, splitBy='unc', image_transforms=transforms_for_image, target_transforms=None, split='val',   eval_mode=False)
        dataset_test  = valid_dataset_class(dataset=opt.dataset, splitBy='unc', image_transforms=transforms_for_image, target_transforms=None, split='test',  eval_mode=False)
        sampler_train = DistributedSampler(dataset_train, shuffle=True,  drop_last=opt.drop_last)
        sampler_val   = DistributedSampler(dataset_val,   shuffle=False, drop_last=opt.drop_last)
        sampler_test  = DistributedSampler(dataset_test,  shuffle=False, drop_last=opt.drop_last)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, collate_fn=multi_collate_refseg, drop_last=opt.drop_last, sampler=sampler_train,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)
        data_loader_val =   torch.utils.data.DataLoader(dataset_val,   batch_size=opt.batch_size, collate_fn=multi_collate_refseg, drop_last=opt.drop_last, sampler=sampler_val,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)
        data_loader_test =  torch.utils.data.DataLoader(dataset_test,  batch_size=opt.batch_size, collate_fn=multi_collate_refseg, drop_last=opt.drop_last, sampler=sampler_test,
                                                        num_workers=opt.num_workers, persistent_workers=opt.persistent_workers, pin_memory=opt.pin_memory, shuffle=False)
        dataloaders_inference = {
            'val': data_loader_val,
            'test': data_loader_test,
        }
        return data_loader_train, dataloaders_inference
    else:
        raise NotImplementedError    


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    input_size, image_size = 224, 224
    transforms_for_image = transforms.Compose([
        transforms.Resize(input_size, image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    # dataset = ReferDataset(dataset='refcoco+', splitBy='unc', image_transforms=transforms_for_image, target_transforms=None, split='testA', eval_mode=False)
    # dataset = ReferDataset(dataset='refcocog', splitBy='umd', image_transforms=transforms_for_image, target_transforms=None, split='val', eval_mode=False)
    # dataset = ReferDataset(dataset='refcoco', splitBy='unc', image_transforms=transforms_for_image, target_transforms=None, split='val', eval_mode=False)
    # dataset = ReferDataset(dataset='refclef', splitBy='unc', image_transforms=transforms_for_image, target_transforms=None, split='val', eval_mode=False)
    dataset = ReferDatasetAll(dataset='refcocog', splitBy='umd', image_transforms=transforms_for_image, target_transforms=None, split='val')
    if False:
        print(len(dataset))
        for i in range(len(dataset)):
            img, target, sentences = dataset.__getitem__(i+999)
            print(i, img.shape, target.shape, img.dtype, target.dtype, target.max(), target.min())
            print(sentences)
            # for sid, sent in enumerate(sentences):
            #     print('%s. %s' % (sid + 1, sent))
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.imshow(target.long().numpy(), alpha=0.5)
            plt.show()
            # input()
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=multi_collate_refseg, num_workers=4)
        for i, data in enumerate(data_loader):
            img, target, sentences = data
            print(img.shape, target.shape, len(sentences))
            # print(target)
            print(sentences)
            
            # print(sentences)
            input()
