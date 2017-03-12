groundtruth_file = 'test.qrel_clean'
answer_file = 'answer'

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
precision = sum([len(q2reld[q].intersection(q2retrd[q]))*1.0/len(q2retrd[q]) for q in q2retrd.keys()]) / N
recall = sum([len(q2reld[q].intersection(q2retrd[q]))*1.0/len(q2reld[q]) for q in q2retrd.keys()]) / N
print("mean precision: {}\nmean recall: {}\nmean F-measure: {}"\
      .format(precision, recall, 2*precision*recall/(precision+recall)))

# MAP@10
import numpy as np

MAP = 0.0
for q in q2retrd.keys():
    n_results = min(10, len(q2retrd[q]))
    avep = np.zeros(n_results)
    for i in range(n_results):
        avep[i:] += q2retrd[q][i] in q2reld[q]
        avep[i] *= (q2retrd[q][i] in q2reld[q]) / (i+1.0)
    MAP += sum(avep) / min(n_results, len(q2reld[q]))
print("MAP@10: {}".format(MAP/N))
