import spacy
import os

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def prepare_text(text):
    text = text.replace("_____", " SEP_TOKEN ")
    tokens = nlp(text)
    joined = " ".join([token.text for token in tokens])
    return joined.replace("SEP_TOKEN", "[SEP]")
