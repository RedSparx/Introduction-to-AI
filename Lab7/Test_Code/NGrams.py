import nltk
from nltk.util import ngrams

def word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s

Sentence = 'one two three four five six seven'
# print(word_grams('one two three four five six seven'.split(' ')))
print(list(ngrams(Sentence.split(' '),3)))