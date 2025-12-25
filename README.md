# entity-linkings

<p align="left">
<i>entity-linkings</i> is a unified library for entity linking.
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
git clone git@github.com:naist-nlp/entity-linkings.git
cd entity-linkings
pip install .

# for uv users
git clone git@github.com:naist-nlp/entity-linkings.git
cd entity-linkings
uv sync
```

## Quick Start
entity-linkigs provides two interfaces: command-line interface (CLI) and Python API.

### CLI
Command-line interface can train/evalate/run Entity Linkings system from command-line.
To create EL system, you must build candidate retriever with ```entitylinkings-train_retrieval```.
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
```

Next, Entity Disambiguation (ED) and End-to-End Entity Linking (EL) systems can trained with ```entitylinkings-train```.
This example is the FEVRY with custom candidate retriever.

```sh
entitylinkings-train \
    --model_type ed \
    --model_id fevry \
    --model_name_or_path google-bert/bert-base-uncased \
    --retriever_id e5bm25 \
    --retriever_model_name_or_path save_model/ \
    --dictionary_id_or_path dictionary.jsonl \
    --train_file train.jsonl \
    --validation_file validation.jsonl \
    --num_candidates 30 \
    --num_train_epochs 2 \
    --train_batch_size 8 \
    --validation_batch_size 16 \
    --output_dir save_fevry/ \
    --config config.yaml \
    --wandb
```

Finally, you can evaluate Retriever or EL systems with ```entitylinkings-eval``` or ```entitylinkings-eval-retrieval```, respectively.
```sh
entitylinkings-eval-retrieval \
    --retriever_id <retriever_id> \
    --model_name_or_path save_model/ \
    --dictionary_id_or_path dictionary.jsonl \
    --test_file test.jsonl \
    --config config.yaml \
    --output_dir result/ \
    --test_batch_size 256 \
    --wandb
```

```sh
entitylinkings-eval \
    --model_type ed \
    --model_id fevry \
    --model_name_or_path save_fevry/ \
    --retriever_id e5bm25 \
    --retriever_model_name_or_path save_model/ \
    --dictionary_id_or_path dictionary.jsonl \
    --test_file test.jsonl \
    --config config.yaml \
    --output_dir result/ \
    --test_batch_size 256 \
    --wandb
```

<!-- #### Interactive Run
```sh
entitylinkings-run \
    --model_id <model_id> \
    --model_name_or_path <model directory> \
    --dictionary_id_or_path <dictionary id> \
    --config config.yaml
``` -->

You can change the arguments (e.g., context length) using configuration file.
The config.yaml with default values can be generated via `entitylinkings-gen-config`.
```sh
entitylinkings-gen-config
```

### Python API
This is the exemple of ChatEL with Zelda Candidate list via API.
Valids IDs for `get_retrievers` and `get_models()` can be found with `get_retriever_ids` and `get_model_ids()` respectively.
```python
from entity_linkings import get_retrievers, get_models, load_dictionary

# Load Dictionary from dictionary_id or local path
dictionary = load_dictionary('zelda')

# Load Candidate Retriever
retriever_cls = get_retrievers('zeldacl')
retriever = retriever_cls(
    dictionary,
    config=retriever_cls.Config()
)

# Setup ED or EL models
model_cls = get_models('chatel')
model = model_cls(
    task='ed'
    retriever=retriever,
    config=model_cls.Config("gpt-4o")
)

# Prediction
sentences = "NAIST is in Ikoma."
spans = [(0, 5)]
predictions = model.predict(sentence, spans, top_k=1)

print("ID: ", predictions[0][0]["id"])
print("Title: ", predictions[0][0]["prediction"])
print("Score: ",  predictions[0][0]["score"])
```

## Available Models
Please refer to the [link](https://colab.research.google.com/drive/1xin1LjNAM7b-Hs0THUbt_-ggj77vOohy?usp=sharing) for instructions on how to run each model.

### Candidate Retriever
* BM25
* ZELDA Candidate List ([Milich and Akbik., 2023](https://github.com/flairNLP/zelda))
* Dual Encoder Model
* Text Embedding Model
* E5+BM25 ([Nakatani et al., 2025](https://aclanthology.org/2025.coling-main.486/))

### Candidate Reranker
* FEVRY ([Févry et al.,2020](https://arxiv.org/abs/2005.14253))
* BLINK ([Wu et al., 2020](https://aclanthology.org/2020.emnlp-main.519/))
* ExtEnD: ([Barba et al., 2022](https://aclanthology.org/2022.acl-long.177))
* FusionED: ([Wang et al., 2024](https://aclanthology.org/2024.naacl-long.363))
* ChatEL ([Ding et al., 2024](https://aclanthology.org/2024.lrec-main.275))

<!-- ### End-to-End Entity Linking
* ReFinED ([Ayoola et al., 2022](https://aclanthology.org/2022.naacl-industry.24/))
* EntQA ([Zhang et al., 2022](https://openreview.net/forum?id=US2rTP5nm_))
* SpEL: ([Shavarani and Sarkar., 2023](https://aclanthology.org/2023.emnlp-main.686))
* FusionED: ([Wang et al., 2024](https://aclanthology.org/2024.naacl-long.363)) -->

## Entity Dictionary
### Available Dictionaries
| dictionary_id | Dataset | Language | Domain |
| :-----  | :-----  | :----- | :------- |
| `kilt` | KILT ([Petroni et al., 2021](https://github.com/facebookresearch/KILT/)) | English | Wikipedia |
| `zelda`| ZELDA ([Milich and Akbik., 2023](https://github.com/flairNLP/zelda)) | English | Wikipedia |
| `zeshel`| ZeshEL ([Logeswaran et al., 2021](https://github.com/lajanugen/zeshel)) | English | Wikia |
<!-- | `mesh` | MeSH | English | UMLS |
| `entrez` | Entrez ([link](https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz)) | English | UMLS |
| `medic` | MEDIC ([link](https://ctdbase.org/downloads)) | English | CTD Disease | -->
<!-- * UMLS is licensed by the National Library of Medicine and requires a free account to download. You can sign up for an account at https://uts.nlm.nih.gov/uts/signup-login.
  * Once your account has been approved, you can download the UMLS metathesaurus at https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html.
* MeSH is derived from UMLS -->

### Custom Entity Dictionary
If you want to use our packages with your custom ontologies, you need to convert to the following format:
```
{
  "id": "000011",
  "name": "NAIST",
  "description": "NAIST is located in Ikoma."
}
```

## Datasets
### Public datasets
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
| `oke-2016` | OKE-2016 ([Nuzzolese et al., 2015](https://github.com/anuzzolese/oke-challenge-2016)) | News | English | Wikipedia | | Unknown* |
| `wned-wiki` | WNED-WIKI ([Guo and Barbosa, 2018](https://github.com/U-Alberta/wned)) | Wikipedia | English | Wikipedia | | Unknown |
| `wned-cweb` | WNED-CWEB ([Guo and Barbosa, 2018](https://github.com/U-Alberta/wned)) | Web | English | Wikipedia | | Apache License 2.0 |
| `unseen` | WikilinksNED Unseen-Mentions ([Onoe and Durrett, 2020](https://github.com/yasumasaonoe/ET4EL)) | News | English | Wikipedia | ✅ | CC-BY 3.0* |
| `tweeki`| Tweeki EL ([Harandizadeh and Singh, 2020](https://ucinlp.github.io/tweeki/)) | Twitter | English | Wikipedia | | Apache License 2.0 |
| `reddit-comments`| Reddit EL ([Botzer et al., 2021](https://doi.org/10.5281/zenodo.3970806)) | Reddit | English | Wikipedia | | CC-BY 4.0 |
| `reddit-posts`| Reddit EL ([Botzer et al., 2021](https://doi.org/10.5281/zenodo.3970806)) | Reddit | English | Wikipedia | | CC-BY 4.0 |
| `shadowlink-shadow` | ShadowLink ([Provatorova et al., 2021](https://huggingface.co/datasets/vera-pro/ShadowLink)) | Wikipedia | English | Wikipedia | | Unknown* |
| `shadowlink-top` | ShadowLink ([Provatorova et al., 2021](https://huggingface.co/datasets/vera-pro/ShadowLink)) | Wikipedia | English | Wikipedia | | Unknown* |
| `shadowlink-tail` | ShadowLink ([Provatorova et al., 2021](https://huggingface.co/datasets/vera-pro/ShadowLink)) | Wikipedia | English | Wikipedia | | Unknown* |
| `zeshel` | Zeshel ([Logeswaran et al., 2021](https://github.com/lajanugen/zeshel)) | Wikia | English | Wikia | ✅ | CC-BY-SA |
| `docred` | Linked-DocRED ([Genest et al., 2023](https://github.com/alteca/Linked-DocRED/)) | News | English | Wikipedia | ✅ | CC-BY 4.0 |
<!-- | `meantime` | MEANTIME ([Minard et al., 2016](http://www.newsreader-project.eu/results/data/wikinews)) | News | Multilingual | Wikipedia | CC-BY 4.0 |
| `voxel` | VoxEL ([Rosales-Méndez et al., 2018](https://users.dcc.uchile.cl/~hrosales/VoxEL.html)) | News | Multilingual | Wikipedia | CC-BY 4.0 | Unknown |
| `mewsli` | Mewsli-9 ([Botha et al., 2020](https://github.com/google-research/google-research/tree/master/dense_representations_for_entity_retrieval/mel)) | News | Multilingual | Wikidata | -->
<!-- | `disease` | NCBI Disease [(Dogan et al, 2014)](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) | PubMed Abstracts | English | MEDIC (MeSH, OMIM)      |
| `gnormplus` | GNormPlus [(Wei et al, 2016)](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus) | PubMed Abstracts | English | Entrez | Unknown |
| `bc5cdr` | BC5CDR [(Li et al, 2016)](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/) | PubMed Abstracts | English | MeSH |
| `medmentions-full` | MedMentions Full [(Mohan and Li, 2019)](https://github.com/chanzuckerberg/MedMentions) | PubMed Abstracts | English | UMLS | CC0 1.0 |
| `medmentions-wt21pv` | MedMentions ST21PV [(Mohan and Li, 2019)](https://github.com/chanzuckerberg/MedMentions) | PubMed Abstracts | English | UMLS | CC0 1.0 |
| `nlm-chem` | NLM Chem [(Islamaj et al, 2021)](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/) | PMC Full-Text | English | MeSH | -->

* Original MSNBC ([Cucerzan, 2007](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.7939/DVN/10968)) is not available due to expiration of the official link. You can download the dataset at [GERBIL official code](https://github.com/dice-group/gerbil/).
* ShadownLink, OKE-{2015,2016} are uncertain to publicly use, but they are provided at official repositories.
* WikilinksNED Unseen-Mentions is created by splitting the [WikilinksNED](https://github.com/yotam-happy/NEDforNoisyText). The WikilinksNED is derived from the [Wikilinks corpus](https://code.google.com/archive/p/wiki-links/downloads), which is made available under CC-BY 3.0.
* The folowing datasests is not publicly available or uncertain. If you want to evaluate these resource, please register the LDC and convert these dataset to our format.
  * AIDA CoNLL-YAGO ([Hoffart et al., 2011](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads)): You must sign the agreement to use [Reuter Corpus](https://trec.nist.gov/data/reuters/reuters.html)
  * TACKBP-2010 ([Ji et al., 2011](https://blender.cs.illinois.edu/paper/kbp2011.pdf)): You must sign Text Analysis Conference (TAC) Knowledge Base Population Evaluation License Agreement.
  <!-- * ACE2004 ([Ratinov et al., 2011]()): You must sign the Linguistic Data Consortium (LDC) Licence Agreeement. -->
  <!-- * AQUAINT ([Milne and Witten, 2008](https://community.nzdl.org/wikification/)): AQUAINT consists 50 randomly selected arcles from the English portion of AQUAINT corpus, which is not publicly available. You can find the information [here](https://tac.nist.gov//data/data_desc.html) -->
* Results for ZeshEL/ZELDA benchmarks (aida-b, tweeki, reddit-*, shadowlink-*, and wned-*) across all models can be found in [the Spreadsheet](https://docs.google.com/spreadsheets/d/1J3xOjhu47N64WkHLJJwP-6KbDB-Bklm0q2FRRqDUW9M/edit?usp=sharing).


### Custom Dataset
If you want to use our packages with the your private dataset, you must convert it to the following format:
```
{
  "id": "doc-001-P1",
  "text": "She graduated from NAIST.",
  "entities": [{"start": 19, "end": 24, "label": ["000011"]}],
}
```
