import json
import os
import pickle
import sys

from datasets import DownloadManager

if __name__ == "__main__":
    output_file = sys.argv[1]

    PATH_TO_REPOSITORY = "https://nlp.informatik.hu-berlin.de/resources/datasets/zelda/zelda_full.zip"
    data_dir = os.path.join(DownloadManager().download_and_extract(PATH_TO_REPOSITORY), "zelda")
    counter = pickle.load(open(os.path.join(data_dir, 'other', 'zelda_mention_entities_counter.pickle'), 'rb'))
    with open(output_file, 'w') as f:
        json.dump(counter, f)
