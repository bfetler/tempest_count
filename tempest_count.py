# word count example from sklearn
# http://scikit-learn.org/stable/modules/feature_extraction.html

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def read_corpus(fname):
    "read corpus from file name"
    corpus = []
    i, lines = 0, ''
    maxx = 2
    with open(fname, 'r') as f:
        for line in f:
#       line = f.readline()
            if len(line) > 2:
#               corpus.append(line.strip())  # old v
                lines += line.strip() + ' '  # new v
                if i % maxx == maxx-1:
                    corpus.append(lines)
                    lines = ''
                i += 1
    return corpus

def count_words(corpus):
    "count words in a corpus (array of strings)"
    vectorizer = CountVectorizer(min_df=1)
    xs = vectorizer.fit_transform(corpus)
    print('type(xs) dir(xs)', type(xs), dir(xs))
    print('dir(vectorizer)', dir(vectorizer))
    print('xs\n%s' % xs)
    print('xs array\n%s' % xs.toarray())
    names = vectorizer.get_feature_names()
    print('feature names (columns) len %d:\n%s' % (len(names), names))
#   print('feature names (columns) len %d:\n%s' % (len(names), names[800:820]))
    print('binary format:', vectorizer.binary)

def count_bigrams(corpus):
    "count bigrams in a corpus"
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), \
        token_pattern=r'\b\w+\b', min_df=1)
    x2 = bigram_vectorizer.fit_transform(corpus)
    print('x2 array\n%s' % (x2.toarray()))
    print('feature names (columns):', bigram_vectorizer.get_feature_names())

def tfid_count_words(corpus):
    "count tfidf words in a corpus (array of strings)"
    vectorizer = TfidfVectorizer(min_df=1)
    xs = vectorizer.fit_transform(corpus)
#   print('type(xs) dir(xs)', type(xs), dir(xs))
#   print('dir(vectorizer)', dir(vectorizer))
#   print('tfid xs\n%s' % xs)
    print('tfid xs array\n%s' % xs.toarray())
    print('feature names (columns):', vectorizer.get_feature_names())
    print('binary format:', vectorizer.binary)

def main():
    corpus = [
        'This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?'
    ]

    fname = 'data/tempest.txt'
    corpus = read_corpus(fname)
#   print('corpus len', len(corpus), '\n', corpus[:10])
    print('corpus len', len(corpus))
#   print(corpus)
    for i in range(10):
        print(corpus[i+40])

    count_words(corpus)

#   count_bigrams(corpus)
#   tfid_count_words(corpus)

if __name__ == '__main__':
    main()

