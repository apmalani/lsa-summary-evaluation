import nltk
import string
import numpy as np

class Evaluater:
    def __init__(self, reference, summary):
        self._reference = reference
        self._summary = summary
        
        nltk.download('punkt')
        nltk.download('stopwords')

    def extract_meaningful_words(self, corpus):
        stop_words = set(nltk.corpus.stopwords('english'))
        words = nltk.tokenize.word_tokenize(corpus)
        meaningful_words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
        return meaningful_words

    def count_sentences(self, corpus):
        sentences = nltk.tokenize.sent_tokenize(corpus)
        return len(sentences)

    def count_words(self, corpus):
        words = nltk.tokenize.word_tokenize(corpus)
        return len(words)

    def create_matrix(self, corpus, word_list):
        sentences = nltk.tokenize.sent_tokenize(corpus)
        words_from_sentences = [nltk.tokenize.word_tokenize(sentence) for sentence in sentences]
        matrix = np.zeros((len(word_list), len(sentences)), dtype = int)

        for i, word in enumerate(word_list):
            for j, sentence in enumerate(words_from_sentences):
                # binary (1) implementation (1 if exists, 0 if not)
                matrix[i, j] = 1 if word in sentence else 0

    def svd(self, matrix):
        pass

    def main_topic_similarity(self):
        pass

    def reduce_normalize_matrices(self, reference_u_matrix, summary_u_matrix):
        pass

    def term_significance_similarity(self):
        pass