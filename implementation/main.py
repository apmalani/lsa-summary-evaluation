import nltk
import string
import numpy as np
import math
import synonyms
from functools import lru_cache

class Evaluater:
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

    def set_ref_sum(self, reference, summary):
        self._reference = reference
        self._summary = summary
        self._rw = self.count_words(self._reference)
        self._sw = self.count_words(self._summary)
        self._dimensions = self.count_sentences(self._reference)

    @staticmethod
    @lru_cache(maxsize=1000)
    def count_sentences(corpus):
        return len(nltk.tokenize.sent_tokenize(corpus))

    @staticmethod
    @lru_cache(maxsize=1000)
    def count_words(corpus):
        return len(nltk.tokenize.word_tokenize(corpus))

    def extract_meaningful_words(self, corpus):
        words = nltk.tokenize.word_tokenize(corpus.lower())
        return [word for word in words if word not in self.stop_words and word not in string.punctuation]

    def create_matrix(self, corpus, word_list, mode="binary"):
        sentences = nltk.tokenize.sent_tokenize(corpus)
        words_from_sentences = [set(nltk.tokenize.word_tokenize(sentence.lower())) for sentence in sentences]
        
        matrix = np.zeros((len(word_list), len(sentences)), dtype=int)
        
        for i, word in enumerate(word_list):
            if mode == "binary":
                matrix[i] = [1 if word in sentence else 0 for sentence in words_from_sentences]
            elif mode == "synonyms":
                syns = set(synonyms.synonym_extractor(word))
                matrix[i] = [1 if (word in sentence or syns.intersection(sentence)) else 0 for sentence in words_from_sentences]
            elif mode == "frequency":
                matrix[i] = [sentence.count(word) for sentence in sentences]
            else:
                raise ValueError("mode not recognized")
        
        return matrix

    @staticmethod
    def svd(matrix):
        return np.linalg.svd(matrix, full_matrices=False)

    @staticmethod
    def main_topic_similarity(reference_u_matrix, summary_u_matrix):
        c = np.dot(reference_u_matrix[:, 0], summary_u_matrix[:, 0])
        return np.arccos(np.clip(c, -1.0, 1.0))

    def reduce_normalize_matrices(self, reference_u_matrix, summary_u_matrix, reference_s_matrix, summary_s_matrix):
        p = min((self._sw / self._rw) * 100, 100)
        r = math.ceil((p / 100) * self._dimensions)

        reduced_u_reference = reference_u_matrix[:, :r]
        reduced_u_summary = summary_u_matrix[:, :r]

        reduced_s_reference = np.square(reference_s_matrix[:r])
        reduced_s_summary = np.square(summary_s_matrix[:r])

        # Ensure both matrices have the same number of columns
        max_cols = max(reduced_u_reference.shape[1], reduced_u_summary.shape[1])
        reduced_u_reference = np.pad(reduced_u_reference, ((0, 0), (0, max_cols - reduced_u_reference.shape[1])))
        reduced_u_summary = np.pad(reduced_u_summary, ((0, 0), (0, max_cols - reduced_u_summary.shape[1])))

        # Ensure s matrices have the same length as the number of columns in u matrices
        reduced_s_reference = np.pad(reduced_s_reference, (0, max_cols - len(reduced_s_reference)))
        reduced_s_summary = np.pad(reduced_s_summary, (0, max_cols - len(reduced_s_summary)))

        b_reference = reduced_u_reference * reduced_s_reference
        b_summary = reduced_u_summary * reduced_s_summary

        l_reference = np.linalg.norm(b_reference, axis=1)
        l_summary = np.linalg.norm(b_summary, axis=1)

        if np.all(l_summary == 0):
            return np.zeros(max_cols), np.zeros(max_cols)

        l_reference = l_reference / np.linalg.norm(l_reference)
        l_summary = l_summary / np.linalg.norm(l_summary)

        return l_reference, l_summary

    @staticmethod
    def term_significance_similarity(l_reference, l_summary):
        c = np.dot(l_reference, l_summary)
        return np.arccos(np.clip(c, -1.0, 1.0))

    def execute_main_topic(self, mode="binary"):
        r_meaningful_words = self.extract_meaningful_words(self._reference)
        r_matrix = self.create_matrix(self._reference, r_meaningful_words, mode)
        s_matrix = self.create_matrix(self._summary, r_meaningful_words, mode)

        r_u, _, _ = self.svd(r_matrix)
        s_u, _, _ = self.svd(s_matrix)

        angle = self.main_topic_similarity(r_u, s_u)
        return 1 - (angle / math.pi)

    def execute_term_sig(self, mode="binary"):
        r_meaningful_words = self.extract_meaningful_words(self._reference)
        r_matrix = self.create_matrix(self._reference, r_meaningful_words, mode)
        s_matrix = self.create_matrix(self._summary, r_meaningful_words, mode)

        r_u, r_s, _ = self.svd(r_matrix)
        s_u, s_s, _ = self.svd(s_matrix)

        l_reference, l_summary = self.reduce_normalize_matrices(r_u, s_u, r_s, s_s)
        angle = self.term_significance_similarity(l_reference, l_summary)
        return 1 - (angle / (math.pi / 2))