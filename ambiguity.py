from nltk.corpus import wordnet
from nltk.corpus import stopwords
import utils
import csv


sense2key = {}
with open("csi/sensekey2csi.tsv", "r") as f:
    for row in f:
        row_entries = row.strip().split("\t")
        sense2key[row_entries[0]] = row_entries[1:]


def wordnet_lesk_ambiguity(context_sentence, ambiguous_word, pos=None, synsets=None):
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


def csi_lesk_ambiguity(context_sentence, ambiguous_word, pos=None, synsets=None):
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
    lesk_results = {}
    for ss in synsets:
        lesk_score = len(context.intersection(ss.definition().split()))
        key = ss.lemmas()[0].key()
        course_senses = sense2key.setdefault(key, [])
        for course_sense in course_senses:
            lesk_results[course_sense] = lesk_results.setdefault(course_sense, 0) + lesk_score
    sorted_course_senses = sorted(lesk_results.items(), key=lambda x: x[1])
    if len(sorted_course_senses) < 2:
        return 0.
    second_highest, highest = sorted_course_senses[-2:]
    second_highest, highest = second_highest[1], highest[1]
    if second_highest == 0:
        return 0.
    return second_highest / highest


def main():
    scoring_fn = wordnet_lesk_ambiguity
    output_extension = "_wordnet_amb"

    #scoring_fn = csi_lesk_ambiguity
    #output_extension = "_csi_amb"

    sw = stopwords.words("english")
    filepaths = ['data/train.tsv', 'data/dev.tsv', 'data/test.tsv']
    for filepath in filepaths:
        with open(filepath, "r") as in_f:
            with open(filepath.replace(".tsv", f"{output_extension}.tsv"), "w") as out_f:
                reader = csv.reader(in_f)
                writer = csv.writer(out_f)
                for row in reader:
                    scores = []
                    text = row[3].replace("_____", " ")
                    text = utils.prepare_text(text).split()
                    for word in text:
                        if word != "[SEP]":
                            word = word.lower()
                        if word in sw:
                            scores.append((word, 0))
                        else:
                            scores.append((word, scoring_fn(text, word)))
                    out_row = row + [scores]
                    writer.writerow(out_row)


if __name__ == "__main__":
    main()
