import logging
import os
from argparse import ArgumentParser, Namespace

from entity_linkings import get_retrievers, load_dataset, load_dictionary
from entity_linkings.trainer import TrainingArguments
from entity_linkings.utils import read_yaml

logger = logging.getLogger(__name__)


def main(args: Namespace) -> None:
    dictionary = load_dictionary(args.dictionary_id_or_path, cache_dir=args.cache_dir)
    dataset_id = args.dataset_id if args.dataset_id else "json"
    if dataset_id != "json":
        dataset = load_dataset(args.dataset_id, cache_dir=args.cache_dir)
    else:
        data_files = {"train": args.train_file}
        if args.validation_file:
            data_files["validation"] = args.validation_file
        dataset = load_dataset("json", data_files=data_files, cache_dir=args.cache_dir)
    if args.remove_nil:
        from entity_linkings.data_utils import filter_nil_entities
        dataset["train"] = filter_nil_entities(dataset["train"], dictionary)
        if "validation" in dataset:
            dataset["validation"] = filter_nil_entities(dataset["validation"], dictionary)

    if args.retriever_config is not None:
        all_config = read_yaml(args.retriever_config)
        model_config = all_config[args.retriever_id.lower()]
        training_config = all_config["training_arguments"]
    else:
        model_config = {}
        training_config = {}

    if args.retriever_model_name_or_path is not None:
        model_config["model_name_or_path"] = args.retriever_model_name_or_path
    training_config["output_dir"] = args.output_dir
    training_config["num_train_epochs"] = args.num_train_epochs
    training_config["per_device_train_batch_size"] = args.train_batch_size
    training_config["per_device_eval_batch_size"] = args.validation_batch_size
    training_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    training_args = TrainingArguments(**training_config)
    if "validation" not in dataset:
        training_args.eval_strategy = 'no'
        training_args.save_strategy = 'no'

    if args.wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "entity_linkings"),
            name=args.retriever_id, tags=["training"]
        )
        wandb.log({
            "model_type": "candidate_retriever",
            "model_name_or_path": args.retriever_model_name_or_path,
            "dictionary_id_or_path": args.dictionary_id_or_path,
            "dataset_id": dataset_id,
            "remove_nil": args.remove_nil,
            "num_hard_negatives": args.num_hard_negatives,
        })
        training_args.report_to = ["wandb"]
        training_args.run_name = args.retriever_id

    retriever_cls = get_retrievers(args.retriever_id)
    retriever = retriever_cls(dictionary, config=retriever_cls.Config(**model_config))
    _ = retriever.train(
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        training_args=training_args,
        num_hard_negatives=args.num_hard_negatives
    )


def cli_main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--retriever_id', '-m', type=str, default=None, help='Name of the model to use.')
    parser.add_argument('--retriever_model_name_or_path', type=str, default=None, help='Name of the model to use.')
    parser.add_argument('--dictionary_id_or_path', '-d', type=str, default=None, help='Path to the entity dictionary file.')
    parser.add_argument('--dataset_id', '-D', type=str, default=None, help='Name of the dataset to use.')
    parser.add_argument('--train_file', type=str, default=None, help='Path to the training dataset file.')
    parser.add_argument('--validation_file', type=str, default=None, help='Path to the validation dataset file.')
    parser.add_argument('--num_hard_negatives', type=int, default=0, help='Number of hard negatives to use during training.')
    parser.add_argument('--num_train_epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument('--validation_batch_size', type=int, default=8, help='Validation batch size.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps.' )
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--remove_nil', action='store_true', default=False, help='Whether to remove nil entities from the dataset.')
    parser.add_argument('--cache_dir', type=str, default=None, help='Path to the cache directory.')
    parser.add_argument('--retriever_config', type=str, default=None, help='YAML-based config file.')
    parser.add_argument('--wandb', action='store_true', default=False, help='Whether to use wandb for logging.')
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
