from implementation import main 
import unittest
import numpy as np

class TestCreatingMatrices(unittest.TestCase):
    examples = "This is a meaningful sentence sentence with literal words.\nIs not a real sentence.\nMeaningful words are awesome."
    
    e = main.Evaluater()
    words = ['sentence', 'actual', 'meaningful']

    def test_creating_matrices(self):
        result = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [1, 0, 1]
        ])

        frequency_result = np.array([
            [2, 1, 0],
            [0, 0, 0],
            [1, 0, 1]
        ])

        self.assertTrue(np.array_equal(self.e.create_matrix(self.examples, self.words, mode = "synonyms"), result))
        self.assertFalse(np.array_equal(self.e.create_matrix(self.examples, self.words, mode = "binary"), result))
        self.assertTrue(np.array_equal(self.e.create_matrix(self.examples, self.words, mode = "frequency"), frequency_result))

if __name__ == '__main__':
    unittest.main()