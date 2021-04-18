import spacy
import os
import logging
import tokenizations

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    try:
        nlp = spacy.load("en_core_web_sm")
    except IndexError:
        print("Warning: failed to load spacy model!")


logger = logging.getLogger(__name__)


def prepare_text(text):  # something happened to the ____ stuff...
    tokens = nlp(text)
    joined = " ".join([token.text for token in tokens])
    return joined.replace("SEP_TOKEN", "[SEP]")


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, token_type_ids, label_id, ambiguity):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.ambiguity = ambiguity


def convert_dataset_to_features(dataset, max_seq_len, tokenizer):
    features = []
    for (ex_index, example) in enumerate(dataset):
        if ex_index % 200 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(dataset)))
        features.append(
            InputFeatures(input_ids=example["text"],
                          input_mask=example["pad_mask"],
                          token_type_ids=example["token_type_ids"],
                          label_id=example["label"],
                          ambiguity=example["ambiguity"]))
    return features
