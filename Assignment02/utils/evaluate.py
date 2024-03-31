from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec  
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import pytrec_eval
import pandas as pd
import numpy as np
import json
import os
    
# import pytrec_eval

import pandas as pd
from nltk.metrics.distance import edit_distance


def tfidf(corpus, name):
    model_path = './model'
    model_name = f'{model_path}/tfidf.{name}'
    if os.path.exists(model_name):
        df_tfidf = pd.read_csv(model_name, header=0)
    else: 
        print(f'Training tfidf model for {name} corpus ...')
        if not os.path.exists(model_path): os.makedirs(model_path)
        vectorizer = TfidfVectorizer()
        data = [" ".join(sent) for sent in corpus]
        tfidf_matrix = vectorizer.fit_transform(data)
        feature_names = vectorizer.get_feature_names_out()
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        df_tfidf.to_csv(model_name, index=False)
    return model_name, df_tfidf

def word2vec(corpus, window_size, vector_size, name):
    model_path = './model'
    model_name = f'{model_path}/word2vec.wordvectors.{name}.{window_size}.{vector_size}'
    if os.path.exists(model_name):
        wv = KeyedVectors.load(model_name, mmap='r')
    else: 
        print(f'Training word2vec model for {name} corpus, window: {window_size}, vector: {vector_size} ...')
        if not os.path.exists(model_path): os.makedirs(model_path)
        model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window_size)
        model.train(corpus, total_examples=1, epochs=1000)
        wv = model.wv
        wv.save(model_name)
    return model_name, wv

def get_topk_tfidf(golden_truth, model_name, model):
    print('Getting top10 words for tfidf ...')
    top10= {}
    top10[model_name] = {}
    vectors = {}
    if os.path.exists(f'./output/{model_name.split("/")[2]}.top10.txt'):
        top10 = read_file(f'./output/{model_name.split("/")[2]}.top10.txt', model_name)
    else: 
        for column_name in model.columns:
            vectors[column_name] = model[column_name].values
        for word in golden_truth.keys():
            top10[model_name][word] = {}
            temp = []
            for w, v in vectors.items():
                if word in model.columns.tolist():
                    vector = model[word].values
                    dot_product = np.dot(vector, v)
                    magnitude1 = np.linalg.norm(vector)
                    magnitude2 = np.linalg.norm(v)
                    try:
                        similarity = dot_product / (magnitude1 * magnitude2)
                    except:
                        similarity = 0
                    temp.append((w, similarity))
            temp = sorted(top10[model_name][word], key=lambda x: x[1], reverse=True)
            top10[model_name][word] = {pair[0]:10 - idx for idx, pair in enumerate(temp[:10])}
        write_to_file(top10[model_name], f'./output/{model_name.split("/")[2]}.top10.txt')
    return top10

def get_topk_word2vec(golden_truth, models):
    print('Getting top10 words for word2vec ...')
    top10= {}
    for name, model in models.items():
        top10[name] = {}
        if os.path.exists(f'./output/{name.split("/")[2]}.top10.txt'):
            top10.update(read_file(f'./output/{name.split("/")[2]}.top10.txt', name))
        else:
            for word in golden_truth.keys():
                try:
                    ms = model.most_similar(word)
                    top10[name][word] = {pair[0]:10 - idx for idx, pair in enumerate(ms[:10])}
                except: continue
            write_to_file(top10[name], f'./output/{name.split("/")[2]}.top10.txt')
    return top10

def read_file(infile, name):
    dictionary = {}
    dictionary[name] = {}
    with open(infile, 'r') as file:
        for line in file:
            inner_dict = {}
            key, value = line.strip().split(':', 1)
            value = value.replace("'", "")
            value = value.replace("}", "")
            value = value.replace("{", "")
            value = value.replace("$10,", "10")
            for item in value.split(','):
                if not ":" in item: continue
                k, v = item.strip().split(':', 1)
                if v.isdigit():
                    inner_dict[k] = int(v)
            dictionary[name][key] = inner_dict
    return dictionary

def write_to_file(dictionary, infile):
    with open(infile, 'w') as f:
        for key, value in dictionary.items():
            f.write(f"{key}:{value}\n")

def test(golden_truth, top10):
    print('Calculating ndcg ...')
    avg_list = []
    evaluator = pytrec_eval.RelevanceEvaluator(golden_truth, {'ndcg'})
    for name, w in top10.items():
        if os.path.exists(f'./output/{name.split("/")[2]}.ndcg'):
             with open(f'./output/{name.split("/")[2]}.ndcg', 'r') as f:
                results = json.load(f)
        else:
            results = evaluator.evaluate(w)
            with open(f'./output/{name.split("/")[2]}.ndcg', 'w') as f:
                json.dump(results, f)
        df = (pd.DataFrame.from_dict(results)).transpose()
        avg_list.append((name, df['ndcg'].mean()))
    return sorted(avg_list, key=lambda x: x[1], reverse=True)[0]

def plot_bar(bc_ndcg_wv, bc_ndcg_tfidf, gc_ndcg_wv, gc_ndcg_tfidf):
    print('Plotting the bar chart ...')
    labels = [bc_ndcg_wv[0].split('/')[2], bc_ndcg_tfidf[0].split('/')[2], gc_ndcg_wv[0].split('/')[2], gc_ndcg_tfidf[0].split('/')[2]]
    values = [bc_ndcg_wv[1], bc_ndcg_tfidf[1], gc_ndcg_wv[1], gc_ndcg_tfidf[1]]
    plt.figure(figsize=(20, 15))
    plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Methods')
    plt.xticks(rotation=45)
    plt.ylabel('nDCG Values')
    plt.title('nDCG Values for Different Methods')
    plt.show()

    


