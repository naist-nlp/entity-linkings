import os
from argparse import ArgumentParser

import yaml

import entity_linkings
from entity_linkings.trainer import TrainingArguments

training_arguments = [
    # Dataloader
    "remove_unused_columns",

    # Training Parameters
    "lr_scheduler_type",
    "warmup_ratio",

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


def write_yaml(output_path: str, content: dict) -> None:
    with open(output_path, 'w') as f:
        yaml.dump(content, f)
        f.write('\n')
        yaml.dump({
            "training_arguments": {
                k: v for k, v in TrainingArguments.__dict__.items()
                if k in training_arguments}
        }, f)


def generate_config(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    retriever_ids = entity_linkings.get_retriever_ids()
    el_ids = entity_linkings.get_el_ids()
    ed_ids = entity_linkings.get_ed_ids()

    for model_id in retriever_ids:
        output_path = os.path.join(output_dir, f"{model_id}.yaml")
        model_cls = entity_linkings.get_retrievers(model_id)
        model_config = model_cls.Config()
        write_yaml(output_path, {f"{model_id}": model_config.__dict__})

    for model_id in el_ids + ed_ids:
        output_path = os.path.join(output_dir, f"{model_id}.yaml")
        model_cls = entity_linkings.get_models(model_id)
        model_config = model_cls.Config()
        write_yaml(output_path, {f"{model_id}": model_config.__dict__})


def cli_main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="configs", help="Output directory for the generated config files.")
    args = parser.parse_args()
    generate_config(args.output_dir)


if __name__ == "__main__":
    cli_main()
