import nltk
import string
import numpy as np
import math

class Evaluater:
    def __init__(self, reference, summary):
        nltk.download('punkt')
        nltk.download('stopwords')
        
        self._reference = reference
        self._summary = summary

        self._rw = self.count_words(self._reference)
        self._sw = self.count_words(self._summary)

        self._dimensions = self.count_sentences(self._summary)

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
        u, s, v = np.linalg.svd(matrix)
        return (u, s, v)

    def main_topic_similarity(self, reference_u_matrix, summary_u_matrix):
        sum = 0
        for i in range(reference_u_matrix.shape[1]):
            sum += np.dot(reference_u_matrix[:,i], summary_u_matrix[:,i])

        angle = math.acos(sum)

        # main topic similarity result
        return angle

    def reduce_normalize_matrices(self, reference_u_matrix, summary_u_matrix, reference_s_matrix, summary_s_matrix):
        p = min((self._sw / self._rw) * 100, 100)
        r = math.ceil((p / 100) * self._dimensions)

        reduced_u_reference = reference_u_matrix[:, :r]
        reduced_u_summary = summary_u_matrix[:, :r]

        reduced_s_reference = reference_s_matrix[:r, :r]
        reduced_s_summary = summary_s_matrix[:r, :r]

        b_reference = np.multiply(reduced_u_reference, reduced_s_reference)

        b_summary = np.multiply(reduced_u_summary, reduced_s_summary)

        l_reference = np.linalg.norm(b_reference, axis = 1)
        l_summary = np.linalg.norm(b_summary, axis = 1)

        l_reference = l_reference / np.linalg.norm(l_reference)
        l_summary = l_summary / np.linalg.norm(l_summary)

        return (l_reference, l_summary)

    def term_significance_similarity(self, l_reference, l_summary):
        c = np.dot(l_reference, l_summary)

        angle = math.acos(c)

        return angle

    def execute(self):
        pass