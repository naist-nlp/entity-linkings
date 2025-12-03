import logging
import re
from argparse import ArgumentParser, Namespace

import torch

from entity_linkings import get_models, get_retrievers, load_dictionary
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
    retriever = retriever_cls(dictionary=dictionary, config=retriever_cls.Config(**retriever_config))

    if args.model_type in ['ed', 'el']:
        if args.model_config is not None:
            model_config = read_yaml(args.model_config).get(args.model_id, {})
        else:
            model_config = {}
        if args.model_name_or_path is not None:
            model_config["model_name_or_path"] = args.model_name_or_path
        model_cls = get_models(args.model_id)
        model = model_cls(retriever=retriever, config=model_cls.Config(**model_config))

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

        if not spans:
            if args.model_type != 'el':
                print("No spans found. Please enclose entity mentions in double square brackets like [[entity]].\n")
                continue

        print('================================')
        if args.model_type == 'retrieval':
            print(f"Model Type: {args.retriever_id}")
        else:
            print(f"Model Type: {args.model_id} with {args.retriever_id}")
        print(f'Input Sentence: {sentence}')
        print(f'Entity Mentions: {[sentence[b:e] for b, e in spans]}')
        print('================================')

        if args.model_type == 'retrieval':
            predictions = retriever.predict(sentence, spans, top_k=args.num_candidates)
        else:
            predictions = model.predict(sentence, spans, num_candidates=args.num_candidates)

        for i, span_preds in enumerate(predictions):
            print(f'Span {i+1}: {sentence[spans[i][0]:spans[i][1]]}')
            for j, pred in enumerate(span_preds):
                print(f'  Top {j+1}: {pred["prediction"]} (ID: {pred["id"]}) - Score: {pred["score"]}')
            print()
        else:
            print("No spans found. Please enclose entity mentions in double square brackets like [[entity]].\n")


def cli_main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, help='Type of the model to use. ["retrieval", "ed", "el"]')
    parser.add_argument('--model_id', type=str, default=None, help='Name of the model to use.')
    parser.add_argument('--model_name_or_path', type=str, default=None, help='Name of the model to use.')
    parser.add_argument('--retriever_id', type=str, required=True, help='Name of the retriever model to use.')
    parser.add_argument('--retriever_model_name_or_path', type=str, default=None, help='Name of the retriever model to use.')
    parser.add_argument('--dictionary_id_or_path', type=str, default=None, help='Path to the entity dictionary file.')
    parser.add_argument('--num_candidates', type=int, default=5, help='Number of candidate entities to consider during evaluation.')
    parser.add_argument('--cache_dir', type=str, default=None, help='Path to the cache directory.')
    parser.add_argument('--model_config', type=str, default=None, help='YAML-based config file.')
    parser.add_argument('--retriever_config', type=str, default=None, help='YAML-based retriever config file.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return.')
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
