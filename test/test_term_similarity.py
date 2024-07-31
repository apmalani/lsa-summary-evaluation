from implementation import main
import unittest
import numpy as np

class test_term_similarity(unittest.TestCase):
    e = main.Evaluater(
        reference = "Hello world. Arun was here. Testing 1 2 3. Goodbye world.",
        summary = "Hello world. Arun was here. Testing 1 2 3. Goodbye world."
    )

    def test_term_similarity(self):
        self.assertEqual(self.e.execute_term_sig("synonyms"), 1.0)

    def test_term_unsimilarity(self):
        self.e.set_ref_sum(reference = "Hello world. Arun was here. Testing 1 2 3. Goodbye world.", summary = "okay")
        self.assertEqual(self.e.execute_term_sig("synonyms"), 0.0)

if __name__ == '__main__':
    unittest.main()