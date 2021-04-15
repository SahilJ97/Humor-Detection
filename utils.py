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
    examples = dataset.get_examples()

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 200 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)

        try:
            encoded = tokenizer.encode_plus(
                tokens_a,
                tokens_b,
                add_special_tokens=True,
                max_length=max_seq_len,
                padding='max_length',
                truncation='longest_first',
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_token_type_ids=True
            )
        except:
            print(example.original)
            continue

        input_ids = encoded.input_ids

        # Align ambiguity scores with BERT tokens
        bert_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        alt_tokens, ambiguity_scores = zip(*example.ambiguity)
        bert_to_alt, alt_to_bert = tokenizations.get_alignments(bert_tokens, alt_tokens)
        aligned_scores = [0]*max_seq_len
        for i in range(len(alt_to_bert)):
            for j in alt_to_bert[i]:
                aligned_scores[j] = ambiguity_scores[i]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in bert_tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in encoded.input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in encoded.attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in encoded.token_type_ids]))
            logger.info("ambiguity scores: %s" % " ".join([str(x) for x in aligned_scores]))
            logger.info("label: %d" % example.label)

        features.append(
            InputFeatures(input_ids=encoded.input_ids,
                          input_mask=encoded.attention_mask,
                          token_type_ids=encoded.token_type_ids,
                          label_id=example.label,
                          ambiguity=aligned_scores))
    return features
