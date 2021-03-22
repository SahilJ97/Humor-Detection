# Code adapted from https://www.nltk.org/_modules/nltk/wsd.html

from nltk.corpus import wordnet


def lesk_ambiguity_score(context_sentence, ambiguous_word, pos=None, synsets=None):
    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    lesk_scores = [len(context.intersection(ss.definition().split())) for ss in synsets]
    second_highest, highest = sorted(lesk_scores)[-1:]

    if second_highest == 0:
        return 0.
    return second_highest / highest
