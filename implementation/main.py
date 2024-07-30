import nltk
import string
import numpy as np
import math

class Evaluater:
    def __init__(self, reference = None, summary = None):
        nltk.download('punkt', quiet = True)
        nltk.download('stopwords', quiet = True)
        
        self._reference = reference
        self._summary = summary

        if reference and summary:
            self._rw = self.count_words(self._reference)
            self._sw = self.count_words(self._summary)

            self._dimensions = self.count_sentences(self._summary)

    def extract_meaningful_words(self, corpus):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = nltk.tokenize.word_tokenize(corpus)
        meaningful_words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
        return meaningful_words

    def count_sentences(self, corpus):
        sentences = nltk.tokenize.sent_tokenize(corpus)
        return len(sentences)

    def count_words(self, corpus):
        words = nltk.tokenize.word_tokenize(corpus)
        return len(words)

    def create_matrix(self, corpus, word_list):
        sentences = nltk.tokenize.sent_tokenize(corpus)
        words_from_sentences = [nltk.tokenize.word_tokenize(sentence.lower()) for sentence in sentences]
        matrix = np.zeros((len(word_list), len(sentences)), dtype = int)

        for i, word in enumerate(word_list):
            for j, sentence in enumerate(words_from_sentences):
                # currently binary, can be changed to the other options (synonyms, threshold, actual value, extractive approach)
                matrix[i, j] = 1 if word in sentence else 0

        return matrix

    def svd(self, matrix):
        u, s, v = np.linalg.svd(matrix)
        return (u, s, v)

    def main_topic_similarity(self, reference_u_matrix, summary_u_matrix):
        sum = 0
        for i in range(reference_u_matrix.shape[1]):
            sum += np.dot(reference_u_matrix[:,i], summary_u_matrix[:,i])

        angle = math.acos(sum)

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

    def execute_main_topic(self):
        r_meaningful_words = self.extract_meaningful_words(self._reference)
        s_meaningful_words = self.extract_meaningful_words(self._summary)

        r_matrix = self.create_matrix(self._reference, r_meaningful_words)
        s_matrix = self.create_matrix(self._summary, s_meaningful_words)

        r_svd = self.svd(r_matrix)
        s_svd = self.svd(s_matrix)

        angle = self.main_topic_similarity(r_svd[0], s_svd[0])

        return 1 - (angle / math.pi)

    def execute_term_sig(self):
        r_meaningful_words = self.extract_meaningful_words(self._reference)
        s_meaningful_words = self.extract_meaningful_words(self._summary)

        r_matrix = self.create_matrix(self._reference, r_meaningful_words)
        s_matrix = self.create_matrix(self._summary, s_meaningful_words)

        r_svd = self.svd(r_matrix)
        s_svd = self.svd(s_matrix)

        l_reference, l_summary = self.reduce_normalize_matrices(r_svd[0], s_svd[0], r_svd[1], s_svd[1])

        angle = self.term_significance_similarity(l_reference, l_summary)

        return 1 - (angle / math.pi)
