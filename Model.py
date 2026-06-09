"""
Multimodal Infusion Tuning (MiT) -- model definition for referring segmentation.

The model keeps a frozen LLaMA and a frozen CLIP vision encoder, and injects
(infuses) the global image representation into the key/value of the textual
self-attention and into the feed-forward block of selected LLaMA layers.

Correspondence with the paper (Section 3.1):
  * Eq.(1-3) K/V infusion          -> fc_embedding_km/vm (multiply, I_d) and
                                       fc_embedding_kp/vp (add, I_a)
  * Eq.(4-5) adaptive head rescale -> L' = L + cos_sim(V_t, I); gate = sigmoid(L')
  * Eq.(7)   feed-forward infusion -> fc_embedding_ff
The last-token hidden state is taken as the infused text representation and,
together with multi-level CLIP features, is decoded into the segmentation mask.

This file is written against transformers 4.35.x (see README / requirements.txt).
The four `custom_llama_*_forward` functions re-implement the corresponding
forward passes of `transformers.models.llama.modeling_llama` from that version,
with the infusion operations inserted.
"""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaTokenizer
from transformers.models.llama import LlamaPreTrainedModel, LlamaModel
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
from transformers.utils import logging
from transformers import CLIPModel

from DecoderTF import MyDecoder as DecoderTF
from DecoderCNN import DecoderMy as DecoderCNN
from Config import ModelLLAMAPath, TokenizerPath, ModelCLIPPATH

logger = logging.get_logger(__name__)


def custom_llama_mlp_forward(self, x, custom_stuffs):
    # Feed-forward infusion, Eq.(7): H'_t = H_t * F_f(I)
    if custom_stuffs['current_layer'] in custom_stuffs['integrate_layers']:
        index = custom_stuffs['integrate_layers'].index(custom_stuffs['current_layer'])
        embedding_ff = custom_stuffs['embeddings_ff'][index].view(x.shape[0], 1, -1)  # [bs, 1, inter_dim]
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x) * embedding_ff)
    else:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


def custom_llama_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        custom_stuffs: List = [None],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)    # [bs, head, len, head_dim]
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ============================ MiT infusion ============================
        if custom_stuffs['current_layer'] in custom_stuffs['integrate_layers']:
            index = custom_stuffs['integrate_layers'].index(custom_stuffs['current_layer'])

            embedding_km = custom_stuffs['embeddings_kms'][index]    # multiply transform for K, I^k_d
            embedding_vm = custom_stuffs['embeddings_vms'][index]    # multiply transform for V, I^v_d
            embedding_kp = custom_stuffs['embeddings_kps'][index]    # add transform for K, I^k_a
            embedding_vp = custom_stuffs['embeddings_vps'][index]    # add transform for V, I^v_a
            integrate_type = custom_stuffs['integrate_type']
            learnable_head_mask = custom_stuffs['learnable_head_masks'][index].squeeze(1)    # learnable L, [bs, head]

            embedding_km = embedding_km.view(key_states.shape[0], key_states.shape[1], 1, key_states.shape[3])  # [bs, head, 1, head_dim]
            embedding_vm = embedding_vm.view(key_states.shape[0], key_states.shape[1], 1, key_states.shape[3])
            embedding_kp = embedding_kp.view(key_states.shape[0], key_states.shape[1], 1, key_states.shape[3])
            embedding_vp = embedding_vp.view(key_states.shape[0], key_states.shape[1], 1, key_states.shape[3])

            # Adaptive head-wise rescaling, Eq.(4-5): L' = L + cos_sim(V_t, I); gate = sigmoid(L')
            similarity = torch.cosine_similarity(
                value_states.flatten(start_dim=2),
                embedding_vp.repeat(1, 1, value_states.shape[2], 1).flatten(start_dim=2), dim=-1)
            learnable_head_mask = learnable_head_mask + similarity
            learnable_head_mask = learnable_head_mask.unsqueeze(-1).unsqueeze(-1)    # [bs, head, 1, 1]
            learnable_head_mask = torch.sigmoid(learnable_head_mask)

            if integrate_type == 'B':
                # K^r_t = K_t + (K_t * I^k_d + I^k_a) * sigmoid(L'); V analogously
                key_states = key_states + (key_states * embedding_km + embedding_kp) * learnable_head_mask
                value_states = value_states + (value_states * embedding_vm + embedding_vp) * learnable_head_mask
            else:
                raise NotImplementedError
        # ======================================================================

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def custom_llama_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    custom_stuffs: List = [None],
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        custom_stuffs=custom_stuffs,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states, custom_stuffs)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


def custom_llama_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        custom_stuffs: List = [None]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            custom_stuffs['current_layer'] = idx
            if idx + 1 == custom_stuffs['integrate_layers']:  # cut the gradient (kept from original)
                hidden_states = hidden_states.detach()

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    custom_stuffs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    custom_stuffs=custom_stuffs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LLamaCustom(LlamaPreTrainedModel):
    """Wraps a frozen LlamaModel and patches its forward passes with the MiT infusion."""

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

        new_llama_model_forward = custom_llama_model_forward.__get__(self.model, self.model.__class__)
        setattr(self.model, 'forward', new_llama_model_forward)

        for i in range(len(self.model.layers)):
            new_llama_decoder_forward = custom_llama_decoder_forward.__get__(self.model.layers[i], self.model.layers[i].__class__)
            setattr(self.model.layers[i], 'forward', new_llama_decoder_forward)

            # We modify the attention operation, so the naive (eager) attention is required.
            assert 'LlamaFlashAttention2' not in str(type(self.model.layers[i].self_attn)), 'Set use_flash_attention_2 to False!'
            new_llama_attention_forward = custom_llama_attention_forward.__get__(self.model.layers[i].self_attn, self.model.layers[i].self_attn.__class__)
            setattr(self.model.layers[i].self_attn, 'forward', new_llama_attention_forward)

            new_llama_mlp_forward = custom_llama_mlp_forward.__get__(self.model.layers[i].mlp, self.model.layers[i].mlp.__class__)
            setattr(self.model.layers[i].mlp, 'forward', new_llama_mlp_forward)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        custom_stuffs: List = [None]
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            custom_stuffs=custom_stuffs,
        )
        hidden_states = transformer_outputs[0]

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(hidden_states.device)
            else:
                sequence_lengths = -1

        # Decoder-only LLM: take the last (non-pad) token as the infused text representation.
        pooled_hiddenstates = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

        return SequenceClassifierOutputWithPast(
            logits=pooled_hiddenstates,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class Model(torch.nn.Module):
    def __init__(self, opt, tokenizer: LlamaTokenizer):
        super(Model, self).__init__()
        self.opt = opt
        self.num_class = opt.num_class
        self.image_size = opt.image_size

        # ----- Frozen LLaMA backbone -----
        self.text_dtype = torch.bfloat16 if opt.bfloat16 else torch.float16
        self.llama_config = LlamaConfig(pad_token_id=tokenizer._convert_token_to_id(tokenizer.pad_token), rms_norm_eps=1e-6, layer_norm_eps=1e-6)
        self.llama = LLamaCustom.from_pretrained(ModelLLAMAPath, config=self.llama_config, load_in_8bit=False, torch_dtype=self.text_dtype, use_flash_attention_2=False)

        self.text_size = self.llama_config.hidden_size                      # text hidden size, e.g. 4096
        self.text_intermediate_size = self.llama_config.intermediate_size   # FFN hidden size, e.g. 11008
        self.text_num_key_value_heads = self.llama_config.num_key_value_heads

        # ----- Frozen CLIP vision encoder -----
        self.vision_dtype = self.text_dtype if opt.vision_half else torch.float32
        clip_model = CLIPModel.from_pretrained(ModelCLIPPATH, torch_dtype=self.vision_dtype)
        self.image_encoder = clip_model.vision_model
        self.image_encoder_projection = clip_model.visual_projection
        self.clip_norm = self.opt.clip_norm
        self.vision_size = clip_model.projection_dim        # global feature dim after projection, e.g. 768
        self.vision_feature_size = clip_model.vision_embed_dim   # intermediate feature dim, e.g. 1024

        # ----- MiT infusion parameters -----
        self.integrate_layers = opt.integrate_layers
        self.integrate_type = opt.integrate_type            # only 'B' (the method in the paper) is supported
        self.integrate_ratio = 1 - opt.integrate_ratio

        # Per-layer transforms that map the global image feature into the LLaMA space.
        # *_km/vm are the multiply transforms (I_d), *_kp/vp the add transforms (I_a), *_ff the FFN transform.
        self.fc_embedding_km = torch.nn.ModuleList([torch.nn.Linear(self.vision_size, self.text_size) for _ in range(len(self.integrate_layers))])
        self.fc_embedding_vm = torch.nn.ModuleList([torch.nn.Linear(self.vision_size, self.text_size) for _ in range(len(self.integrate_layers))])
        self.fc_embedding_kp = torch.nn.ModuleList([torch.nn.Linear(self.vision_size, self.text_size) for _ in range(len(self.integrate_layers))])
        self.fc_embedding_vp = torch.nn.ModuleList([torch.nn.Linear(self.vision_size, self.text_size) for _ in range(len(self.integrate_layers))])
        self.learnable_head_mask = torch.nn.Parameter(torch.ones(len(self.integrate_layers), self.text_num_key_value_heads))    # learnable L
        self.fc_embedding_ff = torch.nn.ModuleList([torch.nn.Linear(self.vision_size, self.text_intermediate_size) for _ in range(len(self.integrate_layers))])

        # ----- Lightweight segmentation decoder -----
        self.decoder_type = opt.decoder_type   # 'TF' (transformer/FiLM, CLIPSeg-style) or 'CNN' (U-Net style)
        self.decoder_levels = opt.decoder_levels
        if self.decoder_type == 'TF':
            self.decoder = DecoderTF(conditional_layer=0, text_dim=self.text_size, reduce_dim=opt.decoder_reduce_dim, patch_size=self.image_encoder.embeddings.patch_size,
                            extract_layers=self.decoder_levels, num_attention_heads=opt.decoder_heads, attention_dropout=opt.decoder_dropout, intermediate_size=opt.decoder_inter_size,
                            vision_hidden_dim=self.vision_feature_size, num_class=self.num_class, init_factor=1.0, init_num_encoder_layers=len(self.image_encoder.encoder.layers))
        elif self.decoder_type == 'CNN':
            self.decoder = DecoderCNN(vision_feature_size=self.vision_feature_size, n_classes=self.num_class)
        else:
            raise NotImplementedError

    # images: [bs, 3, h, w]  text_input_ids/attention_mask: [bs, len]
    def forward(self, images, text_input_ids, text_attention_mask):
        # ----- Extract vision features -----
        with torch.autocast('cuda', dtype=self.vision_dtype):
            vision_output = self.image_encoder(pixel_values=images, output_hidden_states=True)
            _, vision_pooled_output, vision_hidden_states = vision_output[0], vision_output[1], vision_output[2]
            vision_pooled_output = self.image_encoder_projection(vision_pooled_output)
        vision_pooled_output = vision_pooled_output.float()     # global image representation I, [bs, vision_size]
        if self.clip_norm:
            vision_pooled_output = vision_pooled_output / vision_pooled_output.norm(p=2, dim=-1, keepdim=True)

        # Project the global image feature into per-layer infusion embeddings.
        embedding_kms, embedding_kps, embedding_vms, embedding_vps = [], [], [], []
        for layer_km, layer_kp, layer_vm, layer_vp in zip(self.fc_embedding_km, self.fc_embedding_kp, self.fc_embedding_vm, self.fc_embedding_vp):
            embedding_kms.append(layer_km(vision_pooled_output))
            embedding_kps.append(layer_kp(vision_pooled_output))
            embedding_vms.append(layer_vm(vision_pooled_output))
            embedding_vps.append(layer_vp(vision_pooled_output))
        vision_hidden_states = [h.float() for h in vision_hidden_states]

        batch_size = images.shape[0]
        embeddings_ff = [layer_ff(vision_pooled_output) for layer_ff in self.fc_embedding_ff]
        learnable_head_masks = list(self.learnable_head_mask.expand(batch_size, -1, -1).split(1, dim=1))

        custom_stuffs = {
            'vision_pooled_output': vision_pooled_output,
            'embeddings_kms': embedding_kms,
            'embeddings_vms': embedding_vms,
            'embeddings_kps': embedding_kps,
            'embeddings_vps': embedding_vps,
            'embeddings_ff': embeddings_ff,
            'learnable_head_masks': learnable_head_masks,
            'integrate_layers': self.integrate_layers,
            'integrate_type': self.integrate_type,
            'integrate_ratio': self.integrate_ratio,
            'current_layer': 0
        }

        # ----- Extract text features and infuse vision into the LLaMA -----
        with torch.autocast('cuda', dtype=self.text_dtype):
            text_output = self.llama(input_ids=text_input_ids, attention_mask=text_attention_mask, output_hidden_states=True, past_key_values=None, custom_stuffs=custom_stuffs)
        text_last_hidden_state = text_output['logits'].float()      # last-token infused text feature, [bs, text_size]

        # ----- Decode the segmentation mask -----
        vision_decoder_features = [vision_hidden_states[i] for i in self.decoder_levels]
        if self.decoder_type == 'TF':
            decoder_outputs = self.decoder(hidden_states=vision_decoder_features, conditional_embeddings=text_last_hidden_state, output_attentions=True, output_hidden_states=True)
            results = decoder_outputs[0]
        else:
            results = self.decoder(text_last_hidden_state, vision_decoder_features)
        results = F.interpolate(results, [self.image_size, self.image_size], mode='bilinear', align_corners=True)

        # Extra tensors (returned for logging/inspection by the Solver).
        return [results, vision_pooled_output.detach().cpu(), text_last_hidden_state.detach().cpu(),
                vision_decoder_features[0].detach().cpu(), embedding_kms[0].detach().cpu(), embedding_kps[0].detach().cpu()]
