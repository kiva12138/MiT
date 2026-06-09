"""
Lightweight transformer/FiLM segmentation decoder (CLIPSeg-style).

It takes the infused last-token text feature as a conditional embedding (via FiLM
modulation) and several intermediate CLIP feature maps, and produces the mask
logits. This is the default decoder used in the paper for referring segmentation.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class MyCLIPAttention(nn.Module):
    def __init__(self, hidden_size=64, num_attention_heads=4, attention_dropout=0.0):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}")

        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}")
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}")

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class MyCLIPMLP(nn.Module):
    def __init__(self, hidden_size=64, intermediate_size=2048):
        super().__init__()
        self.activation_fn = ACT2FN['relu']
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MyDecoderLayer(nn.Module):
    def __init__(self, hidden_size=64, num_attention_heads=4, attention_dropout=0.0, intermediate_size=2048):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = MyCLIPAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads, attention_dropout=attention_dropout)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = MyCLIPMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class MyDecoder(nn.Module):
    def __init__(self, conditional_layer=0, text_dim=512, reduce_dim=64, patch_size=16, extract_layers=[3, 6, 9],
                 num_attention_heads=4, attention_dropout=0.0, intermediate_size=2048, vision_hidden_dim=768, num_class=2,
                 init_factor=1.0, init_num_encoder_layers=12):
        super().__init__()

        self.num_class = num_class
        self.conditional_layer = conditional_layer

        # FiLM modulation by the (infused) conditional text embedding.
        self.film_mul = nn.Linear(text_dim, reduce_dim)
        self.film_add = nn.Linear(text_dim, reduce_dim)

        transposed_kernels = (patch_size // 4, patch_size // 4)
        self.transposed_convolution = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=transposed_kernels[0], stride=transposed_kernels[0]),
            nn.ReLU(),
            nn.ConvTranspose2d(reduce_dim // 2, num_class, kernel_size=transposed_kernels[1], stride=transposed_kernels[1]),
        )

        self.reduces = nn.ModuleList([nn.Linear(vision_hidden_dim, reduce_dim) for _ in range(len(extract_layers))])
        self.layers = nn.ModuleList([MyDecoderLayer(hidden_size=reduce_dim, num_attention_heads=num_attention_heads,
                                                    attention_dropout=attention_dropout, intermediate_size=intermediate_size) for _ in range(len(extract_layers))])

        for module in self.modules():
            self._init_weights(module, factor=init_factor, num_encoder_layers=init_num_encoder_layers, reduce_dim=reduce_dim)

    def _init_weights(self, module, factor=1.0, num_encoder_layers=12, reduce_dim=64):
        """Initialize the weights (follows the CLIP initialization scheme)."""
        if isinstance(module, MyCLIPAttention):
            in_proj_std = (module.embed_dim**-0.5) * ((2 * num_encoder_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, MyCLIPMLP):
            in_proj_std = ((reduce_dim**-0.5) * ((2 * num_encoder_layers) ** -0.5) * factor)
            fc_std = (2 * reduce_dim) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
        conditional_embeddings: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        activations = hidden_states

        output = None
        for i, (activation, layer, reduce) in enumerate(zip(activations, self.layers, self.reduces)):
            if output is not None:
                output = reduce(activation) + output
            else:
                output = reduce(activation)

            if i == self.conditional_layer:
                output = self.film_mul(conditional_embeddings) * output.permute(1, 0, 2) + self.film_add(conditional_embeddings)
                output = output.permute(1, 0, 2)    # [bs, len, dim]

            layer_outputs = layer(output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions)
            output = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states += (output,)
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        output = output[:, 1:, :].permute(0, 2, 1).contiguous()     # drop cls token -> [bs, reduce_dim, seq_len]
        size = int(math.sqrt(output.shape[2]))
        batch_size = conditional_embeddings.shape[0]
        output = output.view(batch_size, output.shape[1], size, size)

        logits = self.transposed_convolution(output)

        return tuple(v for v in [logits, all_hidden_states, all_attentions] if v is not None)
