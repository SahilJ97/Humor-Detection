from nltk.corpus import wordnet
from nltk.corpus import stopwords
import utils
import csv


def lesk_ambiguity_score(context_sentence, ambiguous_word, pos=None, synsets=None):
    """Code adapted from https://www.nltk.org/_modules/nltk/wsd.html"""
    if not ambiguous_word.isalpha():
        return 0.
    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return 0.

    lesk_scores = [len(context.intersection(ss.definition().split())) for ss in synsets]
    if len(lesk_scores) < 2:
        return 0.

    second_highest, highest = sorted(lesk_scores)[-2:]

    if second_highest == 0:
        return 0.
    return second_highest / highest


def main():
    sw = stopwords.words("english")
    filepaths = ['data/train.tsv', 'data/dev.tsv', 'data/test.tsv']
    for filepath in filepaths:
        with open(filepath, "r") as in_f:
            with open(filepath.replace(".tsv", "_with_amb.tsv"), "w") as out_f:
                reader = csv.reader(in_f)
                writer = csv.writer(out_f)
                for row in reader:
                    scores = []
                    text = utils.prepare_text(row[3]).split()
                    for word in text:
                        if word != "[SEP]":
                            word = word.lower()
                        if word in sw:
                            scores.append((word, 0))
                        else:
                            scores.append((word, lesk_ambiguity_score(text, word)))
                    out_row = row + scores
                    writer.writerow(out_row)


if __name__ == "__main__":
    main()
