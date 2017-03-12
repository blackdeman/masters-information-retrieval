import string
import math
from itertools import groupby
from operator import itemgetter

import nltk
from nltk.tokenize import word_tokenize
# from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

default_tokenize_func = word_tokenize
default_lemmatizer = None  # WordNetLemmatizer()
default_stemmer = SnowballStemmer("english")
default_stop_words = set(nltk.corpus.stopwords.words('english'))


def parse_file(file, label='W'):
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
    tokens = list()

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

        tokens.append(token)

    tokens.sort()

    return [(key, len(list(group))) for key, group in groupby(tokens)]


class InvertedIndex:
    def __init__(self):
        self.documents_total_length = 0
        self.documents = dict()
        self.index = dict()

        pass

    def add_document(self, doc_id, document):
        self.documents_total_length += len(document)
        self.documents[doc_id] = document

        for token in normalize_text(document):
            if token[0] not in self.index:
                self.index[token[0]] = dict()

            self.index[token[0]][doc_id] = token[1]

        pass

    def search_okapi_bm25(self, query, b=0.75, k1=1.2, limit=10):
        result = []

        average_doc_length = self.documents_total_length / len(self.documents)

        query_tokens = [x[0] for x in normalize_text(query)]

        for doc_id, doc_text in self.documents.items():
            rsv = 0
            for token in query_tokens:
                N = len(self.documents)
                Nt = self.get_document_frequency(token)
                ftd = self.get_term_frequency(token, doc_id)
                Ld = len(doc_text)
                L = self.documents_total_length

                idf = math.log(1 + (N - Nt + 0.5) / (Nt + 0.5))
                tf = ftd * (k1 + 1) / (k1 * ((1 - b) + b * Ld / L) + ftd)
                rsv += idf * tf

            result.append((doc_id, rsv))

        result = sorted(result, key=itemgetter(1), reverse=True)
        return [x[0] for x in result][:limit]

    def get_document_frequency(self, term):
        if term in self.index:
            return len(self.index[term])
        else:
            return 0

    def get_term_frequency(self, term, doc_id):
        if term in self.index:
            if doc_id in self.index[term]:
                return self.index[term][doc_id]
            else:
                return 0
        else:
            return 0


index = InvertedIndex()

for doc_tuple in parse_file('cran.all.1400', 'W'):
    index.add_document(doc_tuple[0], doc_tuple[1])

with open('answer_W', 'w') as output:
    for query_tuple in parse_file('cran.qry'):
        query_id = query_tuple[0]
        for doc_id in index.search_okapi_bm25(query_tuple[1]):
            output.write("{} {}\n".format(query_id, doc_id))
