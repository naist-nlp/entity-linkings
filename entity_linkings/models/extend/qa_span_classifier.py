import json
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForQuestionAnswering


class QASpanClassifier(nn.Module):

    # to suppress an AttributeError when training
    _keys_to_ignore_on_save = None
    def __init__(
            self,
            model_name_or_path: str,
            attention_window: Optional[int] = 64,
            modify_global_attention: Optional[int] = 2
        ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path, attention_window=attention_window)
        if "longformer" in config.name_or_path:
            config = AutoConfig.from_pretrained(model_name_or_path, attention_window=attention_window)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path, config=config)
            self.use_longformer = True
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
            self.use_longformer = False
        self.modify_global_attention = modify_global_attention
        self.config = {
            "model_name_or_path": model_name_or_path,
            "attention_window": attention_window,
            "modify_global_attention": modify_global_attention
        }

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            candidates_offsets: Optional[torch.Tensor] = None,
        ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        if self.use_longformer and self.modify_global_attention > 0:
            if candidates_offsets is None:
                raise ValueError("candidates_offsets must be provided when using longformer with modify_global_attention.")
            global_attention = torch.zeros_like(attention_mask)
            if self.modify_global_attention == 1:
                global_attention[:, 0] = 1  # CLS global attention
                for i, candidate_offsets in enumerate(candidates_offsets):
                    for si, ei in candidate_offsets:
                        global_attention[i, si] = 1
                        global_attention[i, ei] = 1
            elif self.modify_global_attention == 2:
                global_attention = torch.zeros_like(attention_mask)
                first_candidate_starts = [
                    min([si for si, _ in cand_offs])
                    for cand_offs in candidates_offsets
                ]
                for i, fcs in enumerate(first_candidate_starts):
                    global_attention[i, :fcs] = 1
            else:
                raise NotImplementedError
            model_inputs["global_attention_mask"] = global_attention

        if labels is not None:
            start_positions, end_positions = labels[0], labels[1]
            model_inputs["start_positions"] = start_positions
            model_inputs["end_positions"] = end_positions

        outputs = self.model(**model_inputs)  # (bs, seq_length, hidden_size)
        logits = torch.stack([outputs.start_logits, outputs.end_logits], dim=0)
        return (outputs.loss, logits)

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        self.model.resize_token_embeddings(new_num_tokens)

    def save_pretrained(self, save_directory: str) -> None:
        self.model.save_pretrained(save_directory)
        json.dump(self.config, open(os.path.join(save_directory, "model_config.json"), "w"), indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "QASpanClassifier":
        config = json.load(open(os.path.join(load_directory, "model_config.json"), "r", encoding="utf-8"))
        config["model_name_or_path"] = load_directory
        model = cls(**config)
        return model
