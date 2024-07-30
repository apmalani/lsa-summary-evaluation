from main import * 
import unittest

class extract_meaningful_words(unittest.TestCase):
    
    e = Evaluater()

    def test_meaningful_words_test(self):
        
        self.assertEqual(self.e.extract_meaningful_words("testing 123"), ['testing', '123'])
        self.assertEqual(self.e.extract_meaningful_words("this is the second example"), ['second', 'example'])

if __name__ == '__main__':
    unittest.main()