import argparse
from Utils import str2bools, str2ints

def parse_args():
    parser = argparse.ArgumentParser()

    # Names, paths, logs
    parser.add_argument("--task_name", default="test")

    # Data and task parameters
    parser.add_argument("--dataset", default='refcoco', type=str)
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--splitBy", default='unc', type=str, choices=['unc', 'umd', 'google'])
    parser.add_argument("--input_size", default=224, type=int, choices=[224, 336, 1024])
    parser.add_argument("--image_size", default=480, type=int)
    parser.add_argument("--train_with_all", action='store_true')
    parser.add_argument("--test_with_all", action='store_true')
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--persistent_workers", action='store_true')
    parser.add_argument("--pin_memory", action='store_true')
    parser.add_argument("--drop_last", action='store_true')
    parser.add_argument("--max_sentence_length", default=-1, type=int) # If -1, then no limited lengths
    
    # Model parameters
    parser.add_argument("--bfloat16", action='store_true')
    parser.add_argument("--vision_half", action='store_true')
    parser.add_argument("--clip_norm", action='store_true')
    parser.add_argument("--integrate_layers", default='29-30-31', type=str2ints)
    parser.add_argument("--integrate_type", default='B', choices=['A', 'B'])
    parser.add_argument("--integrate_ratio", default=0.3, type=float)
    parser.add_argument("--decoder_type", default='TF', choices=['TF', 'CNN'])
    parser.add_argument("--decoder_levels", default='18-12-6', type=str2ints)
    parser.add_argument("--decoder_reduce_dim", default=64, type=int)
    parser.add_argument("--decoder_heads", default=6, type=int)
    parser.add_argument("--decoder_dropout", default=0.0, type=float)
    parser.add_argument("--decoder_inter_size", default=2048, type=int)
    parser.add_argument("--prompt_use", action='store_true')
    parser.add_argument("--tokenizer_padding_side", default='right', choices=['right', 'left'])
    parser.add_argument("--tokenizer_pad_key", default='eos', choices=['unk', 'eos'])
    parser.add_argument("--print_params", action='store_true')
    
    # Training and optimization
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--loss", default='CE', choices=['CE', 'BCE', 'DICE', 'Focal'])
    parser.add_argument("--gradient_clip", default=1.0, type=float)
    parser.add_argument("--gradient_norm", default=-1.0, type=float)
    parser.add_argument("--epochs_num", default=2, type=int)
    parser.add_argument("--optm", default="Adam", type=str, choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument("--amsgrad", action='store_true') # For AdamW
    parser.add_argument("--learning_rate", default=4e-3, type=float)
    parser.add_argument("--decoder_lr_decay", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup_ratio", default=-1, type=float)
    parser.add_argument("--lr_decrease", default='step', type=str, choices=['multi_step', 'step', 'exp', 'seg'])
    parser.add_argument("--lr_decrease_iter", default='60', type=str) # 50, or 50-75
    parser.add_argument("--lr_decrease_rate", default=0.1, type=float) # 0.1/0.5 for exp
    
    parser.add_argument("--save_best_features", action='store_true')
    parser.add_argument("--print_param", action='store_true')
    parser.add_argument("--check_gradient", action='store_true')
    # local_rank, world_size, gpu_id and device are for each process, will be added after running

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    args = parse_args()
    print(args)
