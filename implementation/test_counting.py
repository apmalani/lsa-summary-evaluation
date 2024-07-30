import main 
import unittest

class TestCountingMethods(unittest.TestCase):
    e = main.Evaluater()
    example = "Hello World. Arun was here. Testing 1 2 3."

    def test_counting_sentences(self):
        self.assertEqual(self.e.count_sentences(self.example), 3)

    def test_counting_words(self):
        self.assertEqual(self.e.count_words(self.example), 12)

if __name__ == '__main__':
    unittest.main()