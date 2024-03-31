import pandas as pd
from nltk.corpus import brown, gutenberg

import nltk
# nltk.download('punkt')
# nltk.download('brown')
# nltk.download('gutenberg')

def load_corpus():
    brown_corpus = [[y.lower() for y in x] for x in brown.sents(categories='news')]
    gutenberg_corpus = [[y.lower() for y in x] for x in gutenberg.sents('austen-emma.txt')]
    return brown_corpus, gutenberg_corpus

def get_similar_words(df, word):
    selected_rows = df[df['word1'] == word]
    similar_words = [(w, df.loc[df['word2'] == w, 'SimLex999'].values[0]) for w in selected_rows['word2'].tolist()]
    return sorted(similar_words, key=lambda x: x[1], reverse=True)
    
def expand_similar_words(df, golden_list):
    new_golden_list = golden_list[:]
    for w in golden_list:
        similar_words = get_similar_words(df, w)
        new_golden_list.extend(similar_words)
    return sorted(new_golden_list, key=lambda x: x[1], reverse=True)

def truncate_similar_words(golden_list):
    return golden_list[:10]

def generate_golden_list(simlex_df):
    golden_truth = {}
    for word in simlex_df['word1'].tolist():
        # Get the similar words for the given word
        similar_words = get_similar_words(simlex_df, word)
        # Expand if less than 10
        if len(similar_words) < 10:
            golden_list = expand_similar_words(simlex_df, similar_words)
        # Truncate if more than 10
        if len(similar_words) > 10:
            golden_list = truncate_similar_words(similar_words)
        # Store the golden list
        golden_truth[word] = {pair[0]: len(golden_list) - idx for idx, pair in enumerate(golden_list)}
    return golden_truth

def load_golden_standard(input):
    return generate_golden_list(pd.read_csv(f'{input}/SimLex-999.txt', delimiter='\t', header=0))
