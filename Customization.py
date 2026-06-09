"""Task-specific glue: prompt template, parameter freezing, loss and forward helpers."""
import torch

from Utils import log_message

# Prompt template that wraps each referring expression (Figure 3 in the paper).
prompt_format = "Below is a text that describes a object. \n Segment the object according to the text.\n\n ### Text:\n{text}\n\n### Segmentation:"


def other_model_operations(model, opt):
    """Operations applied right after the model is created: freeze the backbones."""
    def freeze_params(model):
        # Keep LLaMA and the CLIP image encoder frozen; only the MiT module and decoder are tunable.
        for name, param in model.named_parameters():
            if 'image_encoder' in name or 'llama' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def print_params(model):
        for name, param in model.named_parameters():
            msg = '{}\t{}/{}/{}'.format(name, param.requires_grad, str(param.dtype), str(param.shape))
            log_message(msg, opt.local_rank)

    freeze_params(model)
    if opt.print_params:
        print_params(model)


def get_customized_loss(opt):
    return torch.nn.BCEWithLogitsLoss()


def tokenize_batch_texts(tokenizer, texts, opt):
    if opt.max_sentence_length > 0:
        texts = [sentence.split(' ') for sentence in texts]
        texts = [sentence[:opt.max_sentence_length] for sentence in texts]
        texts = [' '.join(sentence) for sentence in texts]

    if opt.prompt_use:
        texts = [prompt_format.format(text=sentence) for sentence in texts]

    text_input = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True)
    return text_input


def compute_outputs_from_model(model, images, sentences, opt):
    sentences = [s[0] for s in sentences]
    text_input = tokenize_batch_texts(opt.tokenizer, sentences, opt)
    outputs = model(images.cuda(), text_input['input_ids'].cuda(), text_input['attention_mask'].cuda())
    return outputs
