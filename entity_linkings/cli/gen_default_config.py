import os
from argparse import ArgumentParser
from dataclasses import MISSING, fields
from typing import Any

import yaml

import entity_linkings
from entity_linkings.trainer import TrainingArguments

training_arguments = [
    # Dataloader
    "remove_unused_columns",

    # Training Parameters
    "lr_scheduler_type",
    "warmup_steps",

    # Optimizer
    "optim",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",

    # Learning Rate and Weight Decay
    "learning_rate",
    "weight_decay",
    "max_grad_norm",

    # Logging
    "log_level",
    "logging_strategy",
    "logging_steps",
    "report_to",

    # Save
    "save_strategy",
    "save_total_limit",

    # Evaluation
    "eval_strategy",
    "metric_for_best_model",
    "load_best_model_at_end",
    "eval_on_start"
]


def default_training_arguments() -> dict[str, Any]:
    '''The declared defaults for the arguments listed above.

    Read off the dataclass fields rather than TrainingArguments.__dict__, which holds
    only what the subclass restates and so left out report_to, and rather than an
    instance, whose __post_init__ has already turned the strings into enums that
    yaml.safe_load cannot read back.
    '''
    defaults = {}
    for field in fields(TrainingArguments):
        if field.name not in training_arguments:
            continue
        if field.default is not MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not MISSING:
            defaults[field.name] = field.default_factory()
    return defaults


TRAINING_CONFIG_NAME = "training_args.yaml"


def write_yaml(output_path: str, content: dict) -> None:
    with open(output_path, 'w') as f:
        yaml.dump(content, f)


def generate_config(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    retriever_ids = entity_linkings.get_retriever_ids()
    reranker_ids = entity_linkings.get_reranker_ids()

    for model_id in retriever_ids:
        retriever_cls = entity_linkings.get_retrievers(model_id)
        write_yaml(
            os.path.join(output_dir, f"{model_id}.yaml"),
            {f"{model_id}": retriever_cls.Config().__dict__},
        )

    for model_id in reranker_ids:
        reranker_cls = entity_linkings.get_rerankers(model_id)
        write_yaml(
            os.path.join(output_dir, f"{model_id}.yaml"),
            {f"{model_id}": reranker_cls.Config().__dict__},
        )

    # The training arguments do not vary by model, so they get one file of their own
    # rather than a copy appended to every model's.
    write_yaml(
        os.path.join(output_dir, TRAINING_CONFIG_NAME),
        {"training_arguments": default_training_arguments()},
    )


def cli_main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="configs", help="Output directory for the generated config files.")
    args = parser.parse_args()
    generate_config(args.output_dir)


if __name__ == "__main__":
    cli_main()
