import os
import json
import random
import pytrec_eval
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nltk.metrics.distance import edit_distance


def get_topk(misspelled_corpus, dictionary, limit=10):
    top_k_words = {}
    for right, wrong in misspelled_corpus:
        distances = [(word, edit_distance(right, word)) for word in dictionary]
        distances.sort(key=lambda x: x[1])
        top_k_words[right] = (wrong, distances[:limit])
    # print(f'Selected top-k words by MED algorithm: {random.sample(top_k_words, 5)}')
    with open('./output/topk_words.json', 'ab') as file: json.dump(top_k_words, file)
    return top_k_words


def success_at_k(top_k_words, output):
    qrel, results = dict(), dict()
    for word, (label, predictions) in top_k_words.items():
        results[word] = {pred[0]: 3 for pred in predictions[0:1]}
        results[word].update({pred[0]: 2 for pred in predictions[1:5] if pred[0] not in results[word]})
        results[word].update({pred[0]: 1 for pred in predictions[5:10] if pred[0] not in results[word]})
        qrel[word] = {label: 1}
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, {'success_1,5,10'}).evaluate(results))
    df_mean = df.mean(axis=1)
    df_mean.to_csv(f'{output}/evaluation_mean.csv', header=False, index=True)
    df.to_csv(f'{output}/evaluation.csv', header=False, index=True)


def parallel(misspelled_pairs, word_net, limit, cluster_number):
    chunks = np.array_split(np.array(misspelled_pairs), len(misspelled_pairs) / cluster_number)
    top_k_words = Parallel(n_jobs=-1, prefer="processes")(delayed(get_topk)(i, word_net, limit) for i in chunks)
    return top_k_words