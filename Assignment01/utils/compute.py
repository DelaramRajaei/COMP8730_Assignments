import json
import pytrec_eval
import pandas as pd
import multiprocessing as mp
from functools import partial


def calculate_min_edit_distance(s1, s2):
    n = len(s1)
    m = len(s2)
    dp = [[-1 for _ in range(m + 1)] for _ in range(n + 1)]
    return min_distance(s1, s2, n, m, dp)


def min_distance(s1, s2, n, m, dp):
    if n == 0:
        return m
    if m == 0:
        return n

    if dp[n][m] != -1:
        return dp[n][m]

    if s1[n - 1] == s2[m - 1]:
        dp[n][m] = min_distance(s1, s2, n - 1, m - 1, dp)
    else:
        m1 = dp[n - 1][m] if dp[n - 1][m] != -1 else min_distance(s1, s2, n - 1, m, dp)
        m2 = dp[n][m - 1] if dp[n][m - 1] != -1 else min_distance(s1, s2, n, m - 1, dp)
        m3 = dp[n - 1][m - 1] if dp[n - 1][m - 1] != -1 else min_distance(s1, s2, n - 1, m - 1, dp)
        dp[n][m] = 1 + min(m1, m2, m3)

    return dp[n][m]


def write_to_json(top_k_words, output):
    try:
        with open(output, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}
    existing_data.update(top_k_words)
    with open(output, 'w') as file:
        json.dump(existing_data, file, indent=2)


def get_topk(misspelled_corpus, dictionary, limit=10):
    top_k_words = {}
    for right, wrong in misspelled_corpus:
        distances = [(word, calculate_min_edit_distance(right, word)) for word in dictionary]
        distances.sort(key=lambda x: x[1])
        top_k_words[right] = (wrong, distances[:limit])
    write_to_json(top_k_words, './output/topk_words.json')
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


def parallel(misspelled_pairs, word_net, limit, chunk_size):
    chunks = [misspelled_pairs[i:i + chunk_size] for i in range(0, len(misspelled_pairs), chunk_size)]
    pool = mp.Pool()
    top_k_words = []
    top_k_words.append(pool.map(partial(get_topk, dictionary=word_net, limit=limit), chunks))
    return top_k_words
