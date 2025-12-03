import json
import logging
import os
from argparse import ArgumentParser, Namespace

import torch

from entity_linkings import get_retrievers, load_dataset, load_dictionary
from entity_linkings.utils import read_yaml

logger = logging.getLogger(__name__)

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def evaluate(args: Namespace) -> None:
    dictionary = load_dictionary(args.dictionary_id_or_path, cache_dir=args.cache_dir)
    dataset_id = args.dataset_id if args.dataset_id else "json"
    if dataset_id != "json":
        test_dataset = load_dataset(dataset_id, split='test', cache_dir=args.cache_dir)
    else:
        test_dataset = load_dataset("json", data_files={"test": args.test_file}, cache_dir=args.cache_dir)['test']
    if args.remove_nil:
        from entity_linkings.data_utils import filter_nil_entities
        test_dataset = filter_nil_entities(test_dataset, dictionary)

    if args.wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "entity_linkings"),
            name=args.retriever_id, tags=["evaluation"]
        )
        wandb.log({
            "retriever_id": args.retriever_id,
            "dataset_id": dataset_id,
            "dictionary_id_or_path": args.dictionary_id_or_path,
            "model_name_or_path": args.retriever_model_name_or_path,
            "remove_nil": args.remove_nil
        })

    if args.retriever_config is not None:
        retriever_config = read_yaml(args.retriever_config).get(args.retriever_id, {})
    else:
        retriever_config = {}
    if args.retriever_model_name_or_path is not None:
        retriever_config["model_name_or_path"] = args.retriever_model_name_or_path

    retriever_cls = get_retrievers(args.retriever_id)
    model = retriever_cls(dictionary=dictionary, config=retriever_cls.Config(**retriever_config))
    metrics = model.evaluate(test_dataset, batch_size=args.test_batch_size)
    logger.info(f"Evaluation results: {metrics}")
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = f"{args.output_dir}/eval_results.json"
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved evaluation results to {output_path}")
    if args.wandb:
        for key, value in metrics.items():
            wandb.log({key: value})


def cli_main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--retriever_id', type=str, required=True, help='Name of the retriever model to use.')
    parser.add_argument('--retriever_model_name_or_path', type=str, default=None, help='Name of the model to use.')
    parser.add_argument('--dictionary_id_or_path', type=str, default=None, help='Path to the entity dictionary file.')
    parser.add_argument('--dataset_id', type=str, default=None, help='Name of the dataset to use.')
    parser.add_argument('--test_file', type=str, default=None, help='Path to the dataset file.')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--remove_nil', action='store_true', default=False, help='Whether to remove nil entities from the dataset.')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output directory.')
    parser.add_argument("--cache_dir", type=str, default=None, help='Path to the cache directory.')
    parser.add_argument('--retriever_config', type=str, default=None, help='YAML-based config file.')
    parser.add_argument('--wandb', action='store_true', default=False, help='Whether to use wandb for logging.')
    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    cli_main()
