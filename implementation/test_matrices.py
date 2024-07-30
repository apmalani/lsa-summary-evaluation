import main 
import unittest
import numpy as np

class TestCreatingMatrices(unittest.TestCase):
    examples = "This is a meaningful sentence with real words.\nIs not a real sentence.\nMeaningful words are awesome."
    
    e = main.Evaluater()
    words = ['sentence', 'real', 'meaningful']

    def test_creating_matrices(self):
        result = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [1, 0, 1]
        ])

        self.assertTrue(np.array_equal(self.e.create_matrix(self.examples, self.words), result))

if __name__ == '__main__':
    unittest.main()