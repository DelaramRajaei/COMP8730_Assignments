"""
colab_quick_start = https://colab.research.google.com/drive/1NyCn4j8OtPQsmAMty41PN6Na9l6T9q4t

Assignment 02
Delaram Rajaei 110124422
"""
from utils import read_input, evaluate
from itertools import product
import time
import random

def print_samples(data, name):
    random_items = random.sample(data, 3)
    print(f"The length of {name} is {len(data)}")
    print(f"Samples of {name}:")
    [print(f'{item}') for item in random_items]
    print('-'*50)

def run(corpus, name, golden_truth):
    print(f'Loading vectors for {name} corpus...')
    word2vec_models = {}
    for window_size, vector_size in product([1, 2, 5, 10], [10, 50, 100, 300]):
        model_name, model = evaluate.word2vec(corpus, window_size, vector_size, name)
        word2vec_models[model_name] = model
    topk_word2vec = evaluate.get_topk_word2vec(golden_truth, word2vec_models)
    ndcg_wv = evaluate.test(golden_truth, topk_word2vec)        
    model_name, model = evaluate.tfidf(corpus, name)
    topk_tfidf = evaluate.get_topk_tfidf(golden_truth, model_name, model)
    ndcg_tfidf = evaluate.test(golden_truth, topk_tfidf)        
    return ndcg_wv, ndcg_tfidf


if __name__ == '__main__':
    # Definig input and output 
    input = './SimLex-999'

    # Loading simlex
    simlex = read_input.load_golden_standard(input=input)    

    # Loading corpora
    bc, gc = read_input.load_corpus()
    print_samples(bc, 'brown corpus')
    print_samples(gc, 'gutenberg corpus')

    start = time.time()
    bc_ndcg_wv, bc_ndcg_tfidf = run(bc, 'brown', simlex)
    gc_ndcg_wv, gc_ndcg_tfidf = run(gc, 'gutenberg', simlex)

    evaluate.plot_bar(bc_ndcg_wv, bc_ndcg_tfidf, gc_ndcg_wv, gc_ndcg_tfidf)
    

    print(f'Runtime of the pipline: {time.time() - start}')