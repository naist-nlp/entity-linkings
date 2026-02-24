import logging
from argparse import ArgumentParser, Namespace

from entity_linkings import get_retrievers, load_dictionary
from entity_linkings.utils import read_yaml

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def build_index(args: Namespace) -> None:
    dictionary = load_dictionary(args.dictionary_id_or_path, cache_dir=args.cache_dir)
    if args.retriever_config is not None:
        retriever_config = read_yaml(args.retriever_config).get(args.retriever_id, {})
    else:
        retriever_config = {}
    if args.retriever_model_name_or_path is not None:
        retriever_config["model_name_or_path"] = args.retriever_model_name_or_path

    retriever_cls = get_retrievers(args.retriever_id)
    model = retriever_cls(dictionary=dictionary, config=retriever_cls.Config(**retriever_config))
    model.retriever.save_index(args.output_dir, ensure_ascii=False)


def cli_main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--retriever_id', type=str, required=True, help='Name of the retriever model to use.')
    parser.add_argument('--retriever_model_name_or_path', type=str, default=None, help='Name of the model to use.')
    parser.add_argument('--retriever_config', type=str, default=None, help='YAML-based config file.')
    parser.add_argument('--dictionary_id_or_path', type=str, default=None, help='Path to the entity dictionary file.')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output directory.')
    parser.add_argument("--cache_dir", type=str, default=None, help='Path to the cache directory.')
    args = parser.parse_args()
    build_index(args)

if __name__ == "__main__":
    cli_main()
