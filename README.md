# entity-linkings

<p align="left">
<i>entity-linkings</i> is an unified library for entity linking.
</p>

<p align="left">
<a href="https://pypi.org/project/entity_linkings"><img alt="PyPi" src="https://img.shields.io/pypi/v/entity_linkings"></a>
<a href="https://github.com/naist-nlp/entity-linkings/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/naist-nlp/entity-linkings"></a>
<a href=""><img src="https://github.com/naist-nlp/entity-linkings/actions/workflows/ci.yaml/badge.svg"></a>
</p>


## Instllation
```
# from PyPi
pip install entity-linkings

# from the source
git clone git@github.com:YuSawan/entity_linkings.git
cd entity_linkings
pip install .

# for uv users
git clone git@github.com:YuSawan/entity_linkings.git
cd entity_linkings
uv sync
```

If you intend to use the pipeline system, please download the spaCy en_core_web_sm model beforehand.
```
python -m spacy download en_core_web_sm
```


## Quick Start
entity-linkigs provides two interfaces: command-line interface (CLI) and Python API.

### CLI
Command-line interface enables you to train/evalate/run Entity Linkings system from command-line.
To set up an EL system, you need to train a candidate retriever using ```entitylinkings-train-retrieval``` and then build its index using ```entitylinkings-build-index```. For retrieval models that do not require training, such as BM25 or Prior, you can skip the training step and directly build the index using ```entitylinkings-build-index```.

In this example, e5bm25 can be executed with custom dataset.

```sh
entitylinkings-train-retrieval \
    --retriever_id  e5bm25 \
    --train_file train.jsonl \
    --validation_file validation.jsonl \
    --dictionary_id_or_path dictionary.jsonl \
    --output_dir save_model/ \
    --num_hard_negatives 4 \
    --num_train_epochs 10 \
    --train_batch_size 8 \
    --validation_batch_size 16 \
    --config config.yaml \
    --wandb

entitylinkings-build-index \
    --retriever_id  e5bm25 \
    --retriever_model_name_or_path save_model/ \
    --dictionary_id_or_path dictionary.jsonl \
    --output_dir e5bm25_index/ \
    --retriever_config config.yaml
```

Next, candidate reranker can trained with ```entitylinkings-train-reranker```.
This example is the FEVRY with custom candidate retriever.

```sh
entitylinkings-train-reranker \
    --reranker_id fevry \
    --reranker_model_name_or_path google-bert/bert-base-uncased \
    --retriever_id e5bm25 \
    --retriever_model_name_or_path save_model/ \
    --retriever_index_dir e5bm25_index/ \
    --dictionary_id_or_path dictionary.jsonl \
    --train_file train.jsonl \
    --validation_file validation.jsonl \
    --num_candidates 30 \
    --num_train_epochs 2 \
    --train_batch_size 8 \
    --validation_batch_size 16 \
    --output_dir save_fevry/ \
    --reranker_config config.yaml \
    --wandb
```

Finally, you can evaluate Retriever or Reranker with ```entitylinkings-eval-retrieval``` or ```entitylinkings-eval-reranker```, respectively.
```sh
entitylinkings-eval-retrieval \
    --retriever_id e5bm25 \
    --retriever_model_name_or_path save_model/ \
    --retriever_index_dir e5bm25_index/ \
    --dictionary_id_or_path dictionary.jsonl \
    --test_file test.jsonl \
    --config config.yaml \
    --output_dir result/ \
    --test_batch_size 256 \
    --wandb
```

```sh
entitylinkings-eval-reranker \
    --reranker_id fevry \
    --reranker_model_name_or_path save_fevry/ \
    --retriever_id e5bm25 \
    --retriever_model_name_or_path save_model/ \
    --dictionary_id_or_path dictionary.jsonl \
    --test_file test.jsonl \
    --config config.yaml \
    --output_dir result/ \
    --test_batch_size 256 \
    --wandb
```

You can change the arguments (e.g., context length) using configuration file.
The config.yaml with default values can be generated via `entitylinkings-gen-config`.
```sh
entitylinkings-gen-config
```

### Python API
This is the exemple of ChatEL with Zelda Candidate list via API.
Valids IDs for `get_retrievers` and `get_rerankers()` can be found with `get_retriever_ids` and `get_reranker_ids()` respectively.
```python
from entity_linkings import (
    ELPipeline,
    get_retrievers,
    get_rerankers,
    load_dictionary
)

# Load Dictionary from dictionary_id or local path
dictionary = load_dictionary('zelda')

# Load Candidate Retriever
retriever_cls = get_retrievers('prior')
retriever = retriever_cls(
    dictionary,
    config=retriever_cls.Config(
        model_name_or_path="mention_counter.json"
    )
)

# Load Candidate Reranker (Optional)
reranker_cls = get_rerankers('chatel')
reranker = model_cls(
    dictionary,
    config=model_cls.Config(
        model_name_or_path = "gpt-4o"
    )
)

# Create Pipeline
pipeline = ELPipeline(retriever, reranker)

# Prediction
sentences = "NAIST is in Ikoma."
spans = [(0, 5)]
predictions = pipeline.predict(sentence, spans)

print("ID: ", predictions[0]["id"])
print("Title: ", predictions[0]["prediction"])
print("Score: ",  predictions[0]["score"])
```

## Available Models
### Candidate Retriever
* BM25
* Prior
* Dual Encoder Model ([Wu et al., 2020](https://aclanthology.org/2020.emnlp-main.519/))
* Text Embedding Model
* E5+BM25 ([Nakatani et al., 2025](https://aclanthology.org/2025.coling-main.486/))

### Candidate Reranker
* FEVRY ([Févry et al.,2020](https://arxiv.org/abs/2005.14253))
* Cross Encoder ([Wu et al., 2020](https://aclanthology.org/2020.emnlp-main.519/))
* ExtEnD ([Barba et al., 2022](https://aclanthology.org/2022.acl-long.177))
* FusionED ([Wang et al., 2024](https://aclanthology.org/2024.naacl-long.363))
* ChatEL ([Ding et al., 2024](https://aclanthology.org/2024.lrec-main.275))


## Entity Dictionary
### Available Dictionaries
The following dictionaries can be used directly by specifying the dictionary_id via the CLI. If you are using the Python API, you can download the dictionaries from the [Official Hugging Face Collection](https://huggingface.co/collections/naist-nlp/entity-linkings) using the ``load_dictionary()`` .

| dictionary_id | Dataset | Language | Domain |
| :-----  | :-----  | :----- | :------- |
| `kilt` | KILT ([Petroni et al., 2021](https://github.com/facebookresearch/KILT/)) | English | Wikipedia |
| `zelda`| ZELDA ([Milich and Akbik., 2023](https://github.com/flairNLP/zelda)) | English | Wikipedia |
| `zeshel`| ZeshEL ([Logeswaran et al., 2021](https://github.com/lajanugen/zeshel)) | English | Wikia |

### Custom Entity Dictionary
If you want to use our packages with your custom dictionaries, you need to convert to the following format:
```
{
  "id": "000011",
  "name": "NAIST",
  "description": "NAIST is located in Ikoma."
}
```

## Datasets
### Public datasets
The following datasets can be used directly by specifying the dataset_id via the CLI. If you are using the Python API, you can download datasets from the [Official Hugging Face Collection](https://huggingface.co/collections/naist-nlp/entity-linkings) using the ``load_dataset()`` from the Hugging Face datasets library.

| dataset_id | Dataset | Domain | Language | Ontology | Train | Licence |
| :----- | :----- | :----- | :------- | :------- | :------- | :-------|
| `kilt` | KILT ([Petroni et al., 2021](https://github.com/facebookresearch/KILT/)) | Wikipedia | English | Wikipedia | ✅ | Unknown* |
| `zelda` | ZELDA ([Milich and Akbik., 2023](https://github.com/flairNLP/zelda)) | Wikimedia | English | Wikipedia | ✅ | Unknown* |
| `msnbc` | MSNBC ([Cucerzan, 2007](http://research.microsoft.com/en-us/um/people/silviu/WebAssistant/TestData/)) | News | English | Wikipedia |  | Unknown* |
| `aquaint` | AQUAINT ([Milne and Witten, 2008](https://community.nzdl.org/wikification/)) | News | English | Wikipedia | | Unknown* |
| `ace2004` | ACE2004 ([Ratinov et al, 2011](https://cogcomp.seas.upenn.edu/page/resource_view/4)) | News | English | Wikipedia | | Unknown* |
| `kore50` | KORE50 ([Hoffart et al., 2012](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads)) | News | English | Wikipedia | | CC BY-SA 3.0 |
| `n3-r128` | N3-Reuters-128 ([R̈oder et al., 2014](https://github.com/dice-group/n3-collection)) | News | English | Wikipedia | | GNU AGPL-3.0 |
| `n3-r500` | N3-RSS-500 ([R̈oder et al., 2014](https://github.com/dice-group/n3-collection)) | RSS | English | Wikipedia | | GNU AGPL-3.0 |
| `derczynski` | Derczynski ([Derczynski et al., 2015](https://huggingface.co/datasets/strombergnlp/ipm_nel)) | Twitter | English | Wikipedia | | CC-BY 4.0 |
| `oke-2015` | OKE-2015 ([Nuzzolese et al., 2015](https://github.com/anuzzolese/oke-challenge)) | News | English | Wikipedia | | Unknown* |
| `oke-2016` | OKE-2016 ([Nuzzolese et al., 2015](https://github.com/anuzzolese/oke-challenge-2016)) | News | English | Wikipedia | ✅ | Unknown* |
| `wned-wiki` | WNED-WIKI ([Guo and Barbosa, 2018](https://github.com/U-Alberta/wned)) | Wikipedia | English | Wikipedia | ✅ | Unknown |
| `wned-cweb` | WNED-CWEB ([Guo and Barbosa, 2018](https://github.com/U-Alberta/wned)) | Web | English | Wikipedia | | Apache License 2.0 |
| `unseen` | WikilinksNED Unseen-Mentions ([Onoe and Durrett, 2020](https://github.com/yasumasaonoe/ET4EL)) | News | English | Wikipedia | ✅ | CC-BY 3.0* |
| `tweeki`| Tweeki EL ([Harandizadeh and Singh, 2020](https://ucinlp.github.io/tweeki/)) | Twitter | English | Wikipedia | | Apache License 2.0 |
| `reddit-comments`| Reddit EL ([Botzer et al., 2021](https://doi.org/10.5281/zenodo.3970806)) | Reddit | English | Wikipedia | ✅ | CC-BY 4.0 |
| `reddit-posts`| Reddit EL ([Botzer et al., 2021](https://doi.org/10.5281/zenodo.3970806)) | Reddit | English | Wikipedia | | CC-BY 4.0 |
| `shadowlink-shadow` | ShadowLink ([Provatorova et al., 2021](https://huggingface.co/datasets/vera-pro/ShadowLink)) | Wikipedia | English | Wikipedia | | Unknown* |
| `shadowlink-top` | ShadowLink ([Provatorova et al., 2021](https://huggingface.co/datasets/vera-pro/ShadowLink)) | Wikipedia | English | Wikipedia | | Unknown* |
| `shadowlink-tail` | ShadowLink ([Provatorova et al., 2021](https://huggingface.co/datasets/vera-pro/ShadowLink)) | Wikipedia | English | Wikipedia | | Unknown* |
| `zeshel` | Zeshel ([Logeswaran et al., 2021](https://github.com/lajanugen/zeshel)) | Wikia | English | Wikia | ✅ | CC-BY-SA |
| `docred` | Linked-DocRED ([Genest et al., 2023](https://github.com/alteca/Linked-DocRED/)) | News | English | Wikipedia | ✅ | CC-BY 4.0 |

* Original MSNBC ([Cucerzan, 2007](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.7939/DVN/10968)) is not available due to expiration of the official link. You can download the dataset at [GERBIL official code](https://github.com/dice-group/gerbil/).
* ShadownLink, OKE-{2015,2016} are uncertain to publicly use, but they are provided at official repositories.
* WikilinksNED Unseen-Mentions is created by splitting the [WikilinksNED](https://github.com/yotam-happy/NEDforNoisyText). The WikilinksNED is derived from the [Wikilinks corpus](https://code.google.com/archive/p/wiki-links/downloads), which is made available under CC-BY 3.0.
* The folowing datasests is not publicly available or uncertain. If you want to evaluate these resource, please register the LDC and convert these dataset to our format.
  * AIDA CoNLL-YAGO ([Hoffart et al., 2011](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads)): You must sign the agreement to use [Reuter Corpus](https://trec.nist.gov/data/reuters/reuters.html)
  * TACKBP-2010 ([Ji et al., 2011](https://blender.cs.illinois.edu/paper/kbp2011.pdf)): You must sign Text Analysis Conference (TAC) Knowledge Base Population Evaluation License Agreement.

### Custom Dataset
If you want to use our packages with the your private dataset, you must convert it to the following format:
```
{
  "id": "doc-001-P1",
  "text": "She graduated from NAIST.",
  "entities": [{"start": 19, "end": 24, "label": ["000011"]}],
}
```
