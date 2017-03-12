import string
import nltk
from nltk.tokenize import word_tokenize
# from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict

default_tokenize_func = word_tokenize
default_lemmatizer = None  # WordNetLemmatizer()
default_stemmer = SnowballStemmer("english")
default_stop_words = set(nltk.corpus.stopwords.words('english'))


def parse_file(file, label='T'):
    section_labels = {'I', 'T', 'A', 'B', 'W'}

    section_found = False
    id = 1

    with open(file, 'r') as f:

        document = ''

        for line in f.readlines():

            stipped_line = line.strip()

            if len(stipped_line) == 0:
                continue

            if stipped_line.startswith('.' + label):
                section_found = True
            elif len(stipped_line) >= 2 and stipped_line[0] == '.' and stipped_line[1] in section_labels:
                if section_found:
                    yield (id, document.strip())
                    document = ''
                    id += 1
                    section_found = False

            elif section_found:
                document += ' ' + stipped_line

        if section_found:
            yield (id, document.strip())


def normalize_text(text,
                   tokenize_func=default_tokenize_func,
                   lemmatizer=default_lemmatizer,
                   stemmer=default_stemmer,
                   stop_words=default_stop_words):
    for token in tokenize_func(text):
        if token in string.punctuation:
            continue

        if token in stop_words:
            continue

        if lemmatizer:
            token = lemmatizer.lemmatize(token)

        # todo probably we need to check stop_words here

        if stemmer:
            token = stemmer.stem(token)

        # todo probably we need to check stop_words here

        yield token


class InvertedIndex:
    def __init__(self):
        self.documents = defaultdict(str)
        self.index = defaultdict(list)

        pass

    def add_document(self, docId, document):
        self.documents[docId] = document

        for token in normalize_text(document):
            self.index[token].append(docId)

        pass


index = InvertedIndex()

for doc_tuple in parse_file('cran.all.1400'):
    index.add_document(doc_tuple[0], doc_tuple[1])

print('OK!')
