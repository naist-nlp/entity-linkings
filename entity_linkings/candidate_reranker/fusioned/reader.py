from typing import Any

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Stack


class FusionDecoder(T5ForConditionalGeneration):
    def __init__(self, config: T5Config) -> None:
        super().__init__(config)
        self.wrap_encoder()

    def forward(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            **kwargs: Any
        ) -> tuple[Any, ...]:

        if input_ids is not None:
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return super(FusionDecoder, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            num_beams: int = 3,
            max_new_tokens: int = 200,
            min_length: int = 2
        ) -> torch.Tensor:
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_length
        )

    def wrap_encoder(self, use_checkpoint: bool =False) -> None:
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self) -> None:
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        self.encoder.block = nn.ModuleList(block)

    def load_t5(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint: bool) -> None:
        assert isinstance(self.encoder, EncoderWrapper), "Encoder must be wrapped to set checkpointing."
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: T5Stack, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.main_input_name = encoder.main_input_name
        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> BaseModelOutput:
        bs, total_length = input_ids.size()
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bs * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bs * self.n_passages, passage_length)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return BaseModelOutput(
            last_hidden_state=outputs[0].view(bs, self.n_passages * passage_length, -1),
            hidden_states=outputs[1:],
            attentions=None
        )


class CheckpointWrapper(nn.Module):
    def __init__(self, module: nn.Module, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(
            self,
            hidden_states: dict[str, torch.Tensor],
            attention_mask: torch.Tensor | None = None,
            position_bias: torch.Tensor | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            encoder_decoder_position_bias: torch.Tensor | None = None,
            **kwargs: Any
        ) -> Any:
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs: Any) -> Any:
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_decoder_position_bias,
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                **kwargs,
            )
        return output


def apply_checkpoint_wrapper(t5stack: T5Stack, use_checkpoint: bool) -> None:
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    t5stack.block = nn.ModuleList(block)
