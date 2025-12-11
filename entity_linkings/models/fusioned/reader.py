
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class FusionELReader(nn.Module):
    # to suppress an AttributeError when training
    _keys_to_ignore_on_save = None

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        model_config = AutoConfig.from_pretrained(self.model_name_or_path)
        if model_config.architectures != ['T5ForConditionalGeneration']:
            raise ValueError("FusionELReader currently only supports T5-based models.")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs, cs, length = input_ids.size()
        input_ids = input_ids.view(bs * cs, length)
        attention_mask = attention_mask.view(bs * cs, length)

        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state # (bs * cs, length, hidden_size)
        fusioned_last_hidden_state = last_hidden_state.view(bs, cs * length, -1) # (bs, cs * length, hidden_size)
        fusioned_attention_mask = attention_mask.view(bs, cs * length)

        return fusioned_last_hidden_state, fusioned_attention_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> Seq2SeqLMOutput:
        fusioned_last_hidden_state, fusioned_attention_mask = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        fusion_enc_obj = BaseModelOutput(last_hidden_state=fusioned_last_hidden_state)
        outputs = self.model(
            encoder_outputs=fusion_enc_obj,
            attention_mask=fusioned_attention_mask,
            labels=labels,
        )
        return outputs

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_generation_length: int) -> torch.Tensor:
        fusioned_last_hidden_state, fusioned_attention_mask = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        fusion_enc_obj = BaseModelOutput(last_hidden_state=fusioned_last_hidden_state)

        generated_ids = self.model.generate(
            encoder_outputs=fusion_enc_obj,
            attention_mask=fusioned_attention_mask,
            max_length=max_generation_length,
        )
        return generated_ids

    def save_pretrained(self, save_directory: str) -> None:
        self.model.save_pretrained(save_directory)

    def from_pretrained(self, load_directory: str) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_directory)
