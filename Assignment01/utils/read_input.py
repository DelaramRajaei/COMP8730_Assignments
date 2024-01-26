import random
import nltk
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
        pairs = [line.strip().split() for line in lines]
    return pairs


def load_birkbeck_corpus(input):
    birkbeck_corpus = []
    birkbeck_corpus_infiles = [f'{input}/FAWTHROP1DAT.643', f'{input}/SHEFFIELDDAT.643']
    for f in birkbeck_corpus_infiles:
      birkbeck_corpus.extend(read_birkbeck_files(f))
    print(f'Number of spell cheaking words: {len(birkbeck_corpus)}')
    print(f'Sample of spell cheaking words: {random.sample(birkbeck_corpus, 5)}')
    return birkbeck_corpus