from main import * 
import unittest

class test_topic_similarity(unittest.TestCase):
    
    e = Evaluater("this is the first example", "this is the second example")

    def test_topic_similarity_test(self):
        
        matrixR = self.e.create_matrix("this is the first example", self.e.extract_meaningful_words("this is the first example"))
        uR, sR, vR = self.e.svd(matrixR)
        matrixS = self.e.create_matrix("this is the second example", self.e.extract_meaningful_words("this is the first example"))
        uS, sS, vS = self.e.svd(matrixS)
    
        self.assertEqual(self.e.main_topic_similarity(uR,uS), 0.7853981633974484)

    def test_execute_main_topic_test(self):
        self.assertEqual(self.e.execute_main_topic(),1-(0.7853981633974484/math.pi))


if __name__ == '__main__':
    unittest.main()