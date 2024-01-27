import re
import nltk
import json
import random
from nltk.corpus import wordnet

nltk.download('wordnet')


def load_dictionary():
    dictionary = list(wordnet.words(lang='eng'))
    print(f'Dictionary Lenght: {len(dictionary)}')
    print(f'Sample of dictionary: {random.sample(dictionary, 5)}')
    return dictionary


def read_birkbeck_files(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        pairs = generate_word_pairs(lines)
    return pairs


def generate_word_pairs(words):
    pairs = []
    for i in range(len(words) - 1):
        if words[i][0] == '$':
            base_word = words[i][1:].rstrip('\n')
            substrings = re.findall(r'(?=(\w{2}))', words[i + 1])
            pairs.extend([(base_word, substring) for substring in substrings])
    return pairs


def load_birkbeck_corpus(input):
    birkbeck_corpus = []
    birkbeck_corpus_infiles = [f'{input}/birkbeck.dat']
    for f in birkbeck_corpus_infiles:
        birkbeck_corpus.extend(read_birkbeck_files(f))
    print(f'Number of spell cheaking words: {len(birkbeck_corpus)}')
    print(f'Sample of spell cheaking words: {random.sample(birkbeck_corpus, 5)}')
    return birkbeck_corpus


def read_json(file_path):
    with open(file_path, 'r') as file: json_data = json.load(file)
    return json_data
