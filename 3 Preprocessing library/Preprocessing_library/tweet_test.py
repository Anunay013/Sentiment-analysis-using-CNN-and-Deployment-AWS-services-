"""
Assignment 3 - Unit Tests
"""

import unittest
from tweet import TextPreprocessing


class TestMyModule(unittest.TestCase):
    """
    This class tests the TextPreprocessing class
    """
    def setUp(self):
        return

    def test_clean_text(self):
        """
        Master test function
        """

        tweet = "@skand this (abc) skand@columbia.edu is a random Tweet. don't run 12:"\
        " https://www.google.com⚡️ #columbia"
        processor = TextPreprocessing(2, 1000)
        result = processor.clean_text(tweet)

        expected_result = "(abc) skand@'t run : ⚡️ columbia"

        self.assertEqual(result, expected_result)

    def test_tokenize_text(self):
        """
        Test tokeniser function
        """

        cleaned_tweet = "(abc) skand@'t run : ⚡️ columbia"
        processor = TextPreprocessing(2, 1000)
        result = processor.tokenize_text(cleaned_tweet)

        expected_result = ['(', 'abc', ')', 'skand', '@', "'", 't', 'run', ':', '⚡', '️',\
         'columbia']

        self.assertEqual(result, expected_result)

    def test_replace_token_with_index(self):

        """
        Test replace token with index function
        """

        tokenized = ['(', 'abc', ')', 'skand', '@', "'", 't', 'run', ':', '⚡', '️', 'columbia']
        processor = TextPreprocessing(2, 1000)
        result = processor.replace_token_with_index(tokenized)

        expected_result = [19, 1, 22, 1, 306, 50, 189, 901, 4, 1, 1, 1]

        self.assertEqual(result, expected_result)

    def test_pad_sequence1(self):

        """
        Test pad sequence condition 1 function
        """

        word_embeddings = [19, 1, 22, 1, 306, 50, 189, 901, 4, 1, 1, 1]
        processor = TextPreprocessing(2, 1000)
        result = processor.pad_sequence(word_embeddings)

        expected_result = [19, 1]

        self.assertEqual(result, expected_result)

    def test_pad_sequence2(self):

        """
        Test pad sequence condition 2 function
        """

        word_embeddings = [19, 1, 22, 1, 306, 50]
        processor = TextPreprocessing(10, 1000)
        result = processor.pad_sequence(word_embeddings)

        expected_result = [19, 1, 22, 1, 306, 50, 0, 0, 0, 0]

        self.assertEqual(result, expected_result)

    def test_process_tweet(self):

        """
        Test pipeline function
        """

        tweet = "@skand this (abc) skand@columbia.edu is a random Tweet. don't run 12:"\
        " https://www.google.com⚡️ #columbia"
        processor = TextPreprocessing(2, 1000)
        result = processor.process_tweet(tweet)
        expected_result = [19, 1]
        self.assertEqual(result, expected_result)
