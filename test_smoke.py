"""
Logic smoke-test for MiT (no pretrained weights / dataset required).

It builds *tiny, randomly-initialized* LLaMA and CLIP-shaped modules so that the
MiT infusion math, the two decoders, and the overall data flow can be exercised
on CPU in a few seconds. This validates the code, NOT the trained model quality.

Run:  python test_smoke.py
"""
import torch
from transformers import LlamaConfig

from Model import LLamaCustom
from DecoderTF import MyDecoder as DecoderTF
from DecoderCNN import DecoderMy as DecoderCNN


def build_custom_stuffs(bs, integrate_layers, text_size, inter_size, n_kv_heads):
    """Build the per-layer infusion tensors that Model.forward normally produces from CLIP."""
    n = len(integrate_layers)
    mk = lambda dim: [torch.randn(bs, dim, requires_grad=True) for _ in range(n)]
    return {
        'embeddings_kms': mk(text_size), 'embeddings_vms': mk(text_size),
        'embeddings_kps': mk(text_size), 'embeddings_vps': mk(text_size),
        'embeddings_ff':  mk(inter_size),
        'learnable_head_masks': [torch.randn(bs, 1, n_kv_heads, requires_grad=True) for _ in range(n)],
        'integrate_layers': integrate_layers,
        'integrate_type': 'B',
        'integrate_ratio': 0.7,
        'current_layer': 0,
    }


def test_llama_infusion():
    print('[1/3] LLamaCustom infusion forward/backward ...', end=' ')
    bs, seq, text_size, inter_size = 2, 7, 128, 256
    n_layers, n_heads, n_kv_heads = 4, 4, 4
    integrate_layers = [2, 3]

    cfg = LlamaConfig(hidden_size=text_size, intermediate_size=inter_size, num_hidden_layers=n_layers,
                      num_attention_heads=n_heads, num_key_value_heads=n_kv_heads, vocab_size=320,
                      pad_token_id=0, attn_implementation='eager')
    llama = LLamaCustom(cfg)
    llama.eval()

    input_ids = torch.randint(1, 320, (bs, seq))
    attn_mask = torch.ones(bs, seq, dtype=torch.long)
    custom = build_custom_stuffs(bs, integrate_layers, text_size, inter_size, n_kv_heads)

    out = llama(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True,
                past_key_values=None, custom_stuffs=custom)
    pooled = out['logits']                       # last-token feature, [bs, text_size]
    assert pooled.shape == (bs, text_size), pooled.shape
    assert len(out['hidden_states']) == n_layers + 1

    loss = pooled.float().sum()
    loss.backward()
    # Gradient must flow back into the infusion embeddings.
    assert custom['embeddings_kms'][0].grad is not None
    assert custom['learnable_head_masks'][0].grad is not None
    print('OK   pooled', tuple(pooled.shape))


def test_decoder_tf():
    print('[2/3] DecoderTF forward/backward ...', end=' ')
    bs, text_size, vis_dim, patch = 2, 128, 64, 14
    feat_len = 16 * 16 + 1                         # 16x16 patches + cls token
    dec = DecoderTF(conditional_layer=0, text_dim=text_size, reduce_dim=32, patch_size=patch,
                    extract_layers=[0, 1, 2], num_attention_heads=4, attention_dropout=0.0,
                    intermediate_size=128, vision_hidden_dim=vis_dim, num_class=2,
                    init_factor=1.0, init_num_encoder_layers=4)
    feats = [torch.randn(bs, feat_len, vis_dim) for _ in range(3)]
    cond = torch.randn(bs, text_size, requires_grad=True)
    logits = dec(hidden_states=feats, conditional_embeddings=cond, output_attentions=True, output_hidden_states=True)[0]
    assert logits.shape[0] == bs and logits.shape[1] == 2, logits.shape
    logits.sum().backward()
    assert cond.grad is not None
    print('OK   logits', tuple(logits.shape))


def test_decoder_cnn():
    print('[3/3] DecoderCNN forward/backward ...', end=' ')
    bs, vis_dim = 2, 64
    feat_len = 16 * 16 + 1
    dec = DecoderCNN(n_classes=2, vision_feature_size=vis_dim)
    text_feat = torch.randn(bs, 4096, requires_grad=True)         # DecoderMy expects 4096-dim text feature
    feats = [torch.randn(bs, feat_len, vis_dim) for _ in range(3)]
    logits = dec(text_feat, feats)
    assert logits.shape[0] == bs and logits.shape[1] == 2, logits.shape
    logits.sum().backward()
    assert text_feat.grad is not None
    print('OK   logits', tuple(logits.shape))


if __name__ == '__main__':
    torch.manual_seed(0)
    test_llama_infusion()
    test_decoder_tf()
    test_decoder_cnn()
    print('\nAll smoke tests passed.')
