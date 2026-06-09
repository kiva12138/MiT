import os

####### Paths & Settings #######
# Edit the paths below to point to your local dataset / model weights.

# Root of the referring-segmentation datasets (RefCOCO / RefCOCO+ / RefCOCOg / RefCLEF),
# organized following the standard `refer` API layout (images + refs/instances annotations).
# Defaults to the in-repo `data/` folder; override with an absolute path if needed.
Data_path       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Frozen LLaMA backbone (HuggingFace format). The paper uses LLaMA-2-7B.
ModelLLAMAPath  = r'D:\Models\LLAMA\llama-2-7b-hf'
TokenizerPath   = r'D:\Models\LLAMA\llama-2-7b-hf'

# Frozen CLIP vision encoder (HuggingFace format). The paper uses clip-vit-large-patch14-336.
ModelCLIPPATH   = r'D:\Models\CLIPHF\clip-vit-large-patch14-336'


####### GPUs used for DistributedDataParallel #######
# One entry per process; launch with `torchrun --nproc_per_node=len(CUDA)`.
CUDA = [0, 1]

####### Dataset splits #######
modes = ['train', 'val', 'test', 'testA', 'testB']

####### Supported datasets #######
supported_datasets = ['refclef', 'refcoco', 'refcoco+', 'refcocog']
