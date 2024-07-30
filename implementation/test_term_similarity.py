import main
import unittest
import numpy as np

class test_term_similarity(unittest.TestCase):
    e = main.Evaluater(
        reference = "Hello world. Arun was here. Testing 1 2 3. Goodbye world.",
        summary = "blah world blah blah blah blah."
    )

    def test_term_similarity(self):
        print(self.e.execute_term_sig())


        

if __name__ == '__main__':
    unittest.main()