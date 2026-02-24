import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = False
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1.e-6
    learning_rate: float = 1.e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 0.0
    log_level: str = "info"
    logging_strategy: str = "steps"
    logging_steps: int = 500
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    eval_strategy: str = "epoch"
    metric_for_best_model: str = "loss"
    load_best_model_at_end: bool = True
    eval_on_start: bool = False


class EntityLinkingTrainer(Trainer):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Any]]:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If num_items_in_batch is not passed,

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculating might be slightly inaccurate when performing gradient accumulation.
        """

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # if self.model_accepts_loss_kwargs:
            # kwargs = {}
            # if num_items_in_batch is not None:
            #     kwargs["num_items_in_batch"] = num_items_in_batch
            # inputs = {**inputs, **kwargs}

        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            if labels is None:
                logger.warning(
                    "Trainer: `compute_loss_func` is defined but `labels=None`. "
                    "Your custom loss function will still be called with labels=None. "
                )
            loss, outputs = self.compute_loss_func(inputs, labels)
        else:
            outputs = model(**inputs)
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    num_items_in_batch = self._get_num_items_in_batch([inputs], self.args.device)
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                    )
                loss = loss.detach().mean()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        detached_logits = nested_detach(logits)
        if len(detached_logits) == 1:
            detached_logits = detached_logits[0]

        return (loss, detached_logits, labels)
