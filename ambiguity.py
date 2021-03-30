# Code adapted from https://www.nltk.org/_modules/nltk/wsd.html

from nltk.corpus import wordnet
from nltk.corpus import stopwords

def lesk_ambiguity_score(context_sentence, ambiguous_word, pos=None, synsets=None):
    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    lesk_scores = [len(context.intersection(ss.definition().split())) for ss in synsets]
    second_highest, highest = sorted(lesk_scores)[-2:]

    if second_highest == 0:
        return 0.
    return second_highest / highest

def main():
    sw = stopwords.words("english")

    filepath = 'data/dev.tsv'
    with open(filepath) as fp:
       line = fp.readline()
       while line:
           cnt = 0
           index = 0
           for i in line:
               index += 1
               if i == ",":
                   cnt += 1
               if cnt == 3:
                   sentence = line[index:]
                   print(sentence)

                   for word in sentence.split():
                       if word in sw: #if the word is a stopword
                           print(0)
                       else:
                           print(lesk_ambiguity_score(sentence, word))


                   cnt = -1000 #reset count so nothing will be printed for the same line again

           line = fp.readline() #next line

if __name__ == "__main__":
    main()
