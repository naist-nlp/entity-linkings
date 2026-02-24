import logging
import re
from argparse import ArgumentParser, Namespace

import torch

from entity_linkings import (
    ELPipeline,
    RerankerBase,
    RetrieverBase,
    get_rerankers,
    get_retrievers,
    load_dictionary,
)
from entity_linkings.utils import read_yaml

logger = logging.getLogger(__name__)

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def main(args: Namespace) -> None:
    dictionary = load_dictionary(args.dictionary_id_or_path, cache_dir=args.cache_dir)
    if args.retriever_config is not None:
        retriever_config = read_yaml(args.retriever_config)[args.retriever_id.lower()]
    else:
        retriever_config = {}
    if args.retriever_model_name_or_path is not None:
        retriever_config["model_name_or_path"] = args.retriever_model_name_or_path

    retriever_cls = get_retrievers(args.retriever_id)
    retriever = retriever_cls(dictionary=dictionary, config=retriever_cls.Config(**retriever_config), index_path=args.retriever_index_dir)

    model: RerankerBase|RetrieverBase = retriever
    if args.reranker_id is not None:
        if args.reranker_config is not None:
            reranker_config = read_yaml(args.reranker_config).get(args.reranker_id, {})
        else:
            reranker_config = {}
        if args.reranker_model_name_or_path is not None:
            reranker_config["model_name_or_path"] = args.reranker_model_name_or_path
        reranker_cls = get_rerankers(args.reranker_id)
        reranker = reranker_cls(retriever, config=reranker_cls.Config(**reranker_config))
        model = reranker

    pipeline = ELPipeline(model=model)

    while True:
        sentence = input('Enter: ')
        spans = []
        while True:
            match = re.search(r'\[\[(.*?)\]\]', sentence)
            if not match:
                break
            spans.append((match.start(), match.end()-4))
            mention = sentence[match.start()+2:match.end()-2]
            sentence = sentence[:match.start()] + mention + sentence[match.end():]

        predictions = pipeline.predict(sentence, spans=spans, num_candidates=args.num_candidates)

        print('================================')
        if args.reranker_id is None:
            print(f"Model Type: {args.retriever_id}")
        else:
            print(f"Model Type: {args.reranker_id} with {args.retriever_id}")
        print(f'Input Sentence: {sentence}')
        for pred in predictions:
            pred_entity = dictionary(pred.id)
            print(f'Entity Mentions: {pred.query}')
            print(f'Predicted ID: {pred.id}, Prediction: {pred_entity["name"]}')
        print('================================')


def cli_main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--retriever_id', type=str, required=True, help='Name of the retriever model to use.')
    parser.add_argument('--retriever_model_name_or_path', type=str, default=None, help='Name of the retriever model to use.')
    parser.add_argument('--retriever_index_dir', type=str, default=None, help='Path to the retriever index directory.')
    parser.add_argument('--retriever_config', type=str, default=None, help='YAML-based retriever config file.')
    parser.add_argument('--reranker_id', type=str, default=None, help='Name of the model to use.')
    parser.add_argument('--reranker_model_name_or_path', type=str, default=None, help='Name of the model to use.')
    parser.add_argument('--reranker_config', type=str, default=None, help='YAML-based reranker config file.')
    parser.add_argument('--dictionary_id_or_path', type=str, default=None, help='Path to the entity dictionary file.')
    parser.add_argument('--num_candidates', type=int, default=5, help='Number of candidate entities to consider during evaluation.')
    parser.add_argument('--cache_dir', type=str, default=None, help='Path to the cache directory.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return.')
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
