# SPDX-License-Identifier: Apache-2.0
# Efficient-DLM implementation maintained with the Drex runtime.
# Adapted from Nace.AI's checkpoint configuration_edlm.py and modeling_edlm.py:
# https://huggingface.co/nace-ai/drex-dlm/tree/8827b917a7e1cb037ce4fcef84f814a16158d9ad
# Combined here so loading weights never imports Python from a checkpoint folder.
"""Efficient-DLM encoder using Qwen3 blocks with bidirectional state attention.

Without an explicit attention mask every token can see every other token.
Callers can provide their own mask to isolate question branches.
"""
import torch
from transformers import Qwen3Config, Qwen3PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model


class EfficientDLMConfig(Qwen3Config):
    model_type = "edlm"

    def __init__(
        self,
        block_size=32,
        dlm_arch="encoder",
        dlm_paradigm="bidirectional",
        dlm_type="llada",
        mask_token_id=151662,
        **kwargs,
    ):
        rope_parameters = kwargs.get("rope_parameters")
        super().__init__(**kwargs)
        if isinstance(rope_parameters, dict) and rope_parameters.get("rope_theta") is not None:
            self.rope_theta = rope_parameters["rope_theta"]
        self.block_size = block_size
        self.dlm_arch = dlm_arch
        self.dlm_paradigm = dlm_paradigm
        self.dlm_type = dlm_type
        self.mask_token_id = mask_token_id


class EfficientDLMModel(Qwen3Model):
    config_class = EfficientDLMConfig

    def forward(self, *args, attention_mask=None, **kwargs):
        if attention_mask is None and getattr(self.config, "dlm_paradigm", None) == "bidirectional":
            attention_mask = self._full_mask(*args, **kwargs)
        return super().forward(*args, attention_mask=attention_mask, **kwargs)

    def _full_mask(self, *args, **kwargs):
        # Qwen3Model turns a missing mask into a causal mask. A zero 4D mask is returned as-is.
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        inputs_embeds = kwargs.get("inputs_embeds")
        past = kwargs.get("past_key_values")
        if inputs_embeds is not None:
            batch, query, _ = inputs_embeds.shape
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
        else:
            batch, query = input_ids.shape
            device = input_ids.device
            dtype = next(self.parameters()).dtype
        seen = past.get_seq_length() if past is not None else 0
        return torch.zeros(batch, 1, query, seen + query, dtype=dtype, device=device)


class EfficientDLM(Qwen3PreTrainedModel):
    config_class = EfficientDLMConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = EfficientDLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        # Same position, not the next token. This head is not the decision pointer.
        logits = self.lm_head(out.last_hidden_state)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=out.past_key_values,
            hidden_states=out.last_hidden_state,
        )
