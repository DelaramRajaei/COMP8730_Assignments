"""
colab_quick_start = https://colab.research.google.com/drive/1NyCn4j8OtPQsmAMty41PN6Na9l6T9q4t

Assignment 01
Delaram Rajaei 110124422
"""
from utils import read_input, compute
import random
import time
import os

if __name__ == '__main__':
    # Definig input and output 
    input = './birkbeck_corpus'
    output = './output'
    # Loading Data
    dictionary = read_input.load_dictionary()
    misspelled_corpus = read_input.load_birkbeck_corpus(input=input)

    start = time.time()
    # Get top_k words
    top_k_path = './output/topk_words.json'
    if os.path.exists(top_k_path):
        top_k_words = read_input.read_json(top_k_path)
    else:
        top_k_words = compute.parallel(misspelled_corpus, dictionary, 10, 100)
    # random_sample = {key: top_k_words[key] for key in random.sample(top_k_words.keys(), 5)}
    print(f'5 Selected top-k words by MED algorithm: {dict(list(top_k_words.items())[:5])}')
    # Calculate success@k
    compute.success_at_k(top_k_words, output=output)
    print(f'Runtime of the pipline: {time.time() - start}')


