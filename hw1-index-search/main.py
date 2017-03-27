import string
import math
from itertools import groupby
from operator import itemgetter
from numpy import arange

import nltk
from nltk import defaultdict
from nltk.tokenize import word_tokenize
# from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

default_tokenize_func = word_tokenize
default_lemmatizer = None  # WordNetLemmatizer()
default_stemmer = SnowballStemmer("english")
default_stop_words = set(nltk.corpus.stopwords.words('english'))


def eval_metrics(groundtruth_file, answer_file):
    q2reld = {}
    for line in open(groundtruth_file):
        qid, did = [int(x) for x in line.split()]
        if qid in q2reld.keys():
            q2reld[qid].add(did)
        else:
            q2reld[qid] = set()

    q2retrd = {}
    for line in open(answer_file):
        qid, did = [int(x) for x in line.split()]
        if qid in q2retrd.keys():
            q2retrd[qid].append(did)
        else:
            q2retrd[qid] = []

    N = len(q2retrd.keys())
    precision = sum([len(q2reld[q].intersection(q2retrd[q])) * 1.0 / len(q2retrd[q]) for q in q2retrd.keys()]) / N
    recall = sum([len(q2reld[q].intersection(q2retrd[q])) * 1.0 / len(q2reld[q]) for q in q2retrd.keys()]) / N
    print("mean precision: {}\nmean recall: {}\nmean F-measure: {}" \
          .format(precision, recall, 2 * precision * recall / (precision + recall)))

    # MAP@10
    import numpy as np

    MAP = 0.0
    for q in q2retrd.keys():
        n_results = min(10, len(q2retrd[q]))
        avep = np.zeros(n_results)
        for i in range(n_results):
            avep[i:] += q2retrd[q][i] in q2reld[q]
            avep[i] *= (q2retrd[q][i] in q2reld[q]) / (i + 1.0)
        MAP += sum(avep) / min(n_results, len(q2reld[q]))
    print("MAP@10: {}".format(MAP / N))


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

        if stemmer:
            token = stemmer.stem(token)

        tokens.append(token)

    tokens.sort()

    return [(key, len(list(group))) for key, group in groupby(tokens)]


def calc_rsv_default(N, Nt, ftd, ftq, Ld, L, b, k1, k2):
    idf = math.log(1 + (N - Nt + 0.5) / (Nt + 0.5))
    tf = ftd * (k1 + 1) / (k1 * ((1 - b) + b * Ld / L) + ftd)
    return idf * tf


def calc_rsv_custom_idf(N, Nt, ftd, ftq, Ld, L, b, k1, k2):
    idf = math.log(N / Nt)
    tf = ftd * (k1 + 1) / (k1 * ((1 - b) + b * Ld / L) + ftd)
    return idf * tf


def calc_rsv_general(N, Nt, ftd, ftq, Ld, L, b, k1, k2):
    idf = math.log(1 + (N - Nt + 0.5) / (Nt + 0.5))
    tftd = ftd * (k1 + 1) / (k1 * ((1 - b) + b * Ld / L) + ftd)
    tftq = ftq * (k2 + 1) / (k2 * ftq)
    return idf * tftd * tftq


class InvertedIndex:
    def __init__(self):
        self.documents_total_length = 0
        self.documents = dict()
        self.index = defaultdict(list)

        pass

    def add_document(self, doc_id, document):
        self.documents_total_length += len(document)
        self.documents[doc_id] = document

        # todo maybe sort after add
        for token in normalize_text(document):
            self.index[token[0]].append((doc_id, token[1]))

        pass

    def search_okapi_bm25(self, query, b=0.75, k1=1.2, k2=500, limit=10, rsvfunc=calc_rsv_default, norm_rsv=False):
        result = defaultdict(int)

        N = len(self.documents)
        L = self.documents_total_length / len(self.documents)

        query_terms = dict(normalize_text(query))

        term_idf_sum = 0

        for token in self.index.keys():
            if token in query_terms:
                token_docs = self.index[token]
                Nt = len(token_docs)
                ftq = query_terms[token]
                for doc_tuple in token_docs:
                    doc_id = doc_tuple[0]
                    ftd = doc_tuple[1]
                    Ld = len(self.documents[doc_id])

                    result[doc_id] += rsvfunc(N, Nt, ftd, ftq, Ld, L, b, k1, k2)

                term_idf_sum += math.log(N / Nt)

        result = list(result.items())

        if norm_rsv:
            result = [(x[0], x[1] / term_idf_sum) for x in result]
            pass

        result = sorted(result, key=itemgetter(1), reverse=True)
        return [x[0] for x in result][:limit]

    def get_average_postings_list_length(self):
        l = [len(p) for p in self.index.values()]
        return sum(l) / len(l)

    def get_max_postings_list_length(self):
        l = [len(p) for p in self.index.values()]
        return max(l)

index = InvertedIndex()

for doc_tuple in parse_file('cran.all.1400', 'T'):
    index.add_document(doc_tuple[0], doc_tuple[1])

# for k1 in arange(1.2, 2.1, 0.1):
#     for b in arange(0., 1.1, 0.1):
#         print("------------------------------------------")
#         print("k1 = {}, b = {}".format(k1, b))
with open('answer_T', 'w') as output:
    for query_tuple in parse_file('cran.qry'):
        query_id = query_tuple[0]
        for doc_id in index.search_okapi_bm25(query_tuple[1]):
            output.write("{} {}\n".format(query_id, doc_id))

eval_metrics('qrel_clean', 'answer_T')

print("Average postings list length: {}".format(index.get_average_postings_list_length()))
print("Max postings list length: {}".format(index.get_max_postings_list_length()))
