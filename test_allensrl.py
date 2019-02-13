import unittest
import allen_srl
import filecmp
from allennlp.predictors import Predictor
import os

class TestAllenSRL(unittest.TestCase):


    def test_single_sent(self):

        output_file="path_to_results/test_single_res.txt"
        expected="test_files/expected_single.txt"
        
        result = os.system("python allen_srl.py path_to_test_files/test_files/test_single.txt --output-file test_single_res.txt")
        
        self.assertTrue(filecmp.cmp(expected, output_file))
        
        
        print('ret=',result)

    def test_batch_sent(self):

        output_file="path_to_results/test_batch_res.txt"
        expected="test_files/expected_batch.txt"
        
        result = os.system("python allen_srl.py path_to_test_files/test_files/test_batch.txt --output-file test_batch_res.txt")
        
        self.assertTrue(filecmp.cmp(expected, output_file))
        
        
        print('ret=',result)

    def test_empty_sent(self):

        al = allen_srl.allenSRL()

        
        output_file="path_to_results/test_empty_res.txt"
        expected="test_files/expected_empty.txt"
        
        result = os.system("python allen_srl.py path_to_test_files/test_files/test_empty.txt --output-file test_empty_res.txt")
        
        self.assertTrue(filecmp.cmp(expected, output_file))
        
        
        print('ret=',result)

    def test_bad_sent(self):

        output_file="path_to_results/test_bad_res.txt"
        expected="test_files/expected_bad.txt"
        
        result = os.system("python allen_srl.py path_to_test_files/test_files/test_bad_input.txt --output-file test_bad_res.txt")
        
        self.assertTrue(filecmp.cmp(expected, output_file))
        
        
        print('ret=',result)

    def test_mixed_sent(self):

        output_file="path_to_results/test_mixed_res.txt"
        expected="test_files/expected_mixed.txt"
        
        result = os.system("python allen_srl.py path_to_test_files/test_files/test_mixed.txt --output-file test_mixed_res.txt")
        
        self.assertTrue(filecmp.cmp(expected, output_file))
        
        
        print('ret=',result)

__end__ = '__end__'

if __name__ == '__main__':
    unittest.main()

