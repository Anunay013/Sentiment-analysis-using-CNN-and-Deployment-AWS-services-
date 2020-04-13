"""
Assignmnet 3 - Group 2 - Check unit test for Travis
"""
import re
import os
from zipfile import ZipFile
from tokenizer_fun import TweetTokenizer

class TextPreprocessing:
    """
    This class takes a tweet as an input and returns padded word embeddings
    """

    def __init__(self, max_length_tweet=100, max_length_dictionary=2000000):

        """
        Initialize class
        """
        self.max_length_tweet = max_length_tweet
        self.max_length_dictionary = max_length_dictionary

        # import dictionary
        self.embeddings = []
        self.stopwords = []
        i = 0

        try:
            with open('dictionary.txt', 'r') as file:
                for line in file:
                    word = line.strip()
                    self.embeddings.append(word)
                    i += 1
                    if i >= max_length_dictionary:
                        break

            with open('english_stopwords.txt', 'r') as file:
                for line in file:
                    word = line.strip()
                    self.stopwords.append(word)

        except:
            dictionary_file_path = './Preprocessing_library.zip/dictionary.txt'

            archive_path = os.path.abspath(dictionary_file_path)
            split = archive_path.split(".zip/")
            archive_path = split[0] + ".zip"
            path_inside = split[1]
            archive = ZipFile(archive_path, "r")
            self.embeddings = archive.read(path_inside).decode("utf8").split("\n")
            self.embeddings = self.embeddings[:max_length_dictionary]

            # import stopwords
            stopwords_file_path = './Preprocessing_library.zip/english_stopwords.txt'
            archive_path = os.path.abspath(stopwords_file_path)
            split = archive_path.split(".zip/")
            archive_path = split[0] + ".zip"
            path_inside = split[1]
            archive = ZipFile(archive_path, "r")
            self.stopwords = archive.read(path_inside).decode("utf8").split("\n")

        # define tokenizer
        self.tokenizer = TweetTokenizer()

    def clean_text(self, tweet):

        """
        Clean text
        """

        # lower case
        tweet = tweet.lower()

        # remove links
        tweet = re.sub(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*", '', tweet)

        # replace # to ''
        tweet = re.sub(r"#", '', tweet)

        # remove numbers
        tweet = re.sub(r"[0-9]+", '', tweet)

        # remove stopwords and twitter handles
        tweet_cleaned_list = []
        for tweet_word in tweet.split(" "):
            if (tweet_word not in self.stopwords) & (~tweet_word.startswith('@')):
                tweet_cleaned_list.append(tweet_word)

        # join words
        tweet_cleaned = " ".join(tweet_cleaned_list)

        return tweet_cleaned

    def tokenize_text(self, tweet_cleaned):

        """
        Tokenize text
        """

        # tokenize
        tokenized_words = self.tokenizer.tokenize(tweet_cleaned)

        return tokenized_words

    def replace_token_with_index(self, tokenized):

        """
        Replace token with embeddings
        """

        word_embeddings = []
        for token in tokenized:
            try:
                embedding = self.embeddings.index(token)
                word_embeddings.append(embedding)
            except ValueError:
                embedding = self.embeddings.index('<unknown>')
                word_embeddings.append(embedding)

        return word_embeddings

    def pad_sequence(self, word_embeddings):

        """
        Pad embeddings for model
        """

        # if word_embeddings > max_length_tweet
        if len(word_embeddings) > self.max_length_tweet:
            word_embeddings_pad = word_embeddings[:self.max_length_tweet]

        # if word_embeddings == max_length_tweet
        elif len(word_embeddings) == self.max_length_tweet:
            word_embeddings_pad = word_embeddings

        # if word_embeddings < max_length_tweet
        else:
            diff = self.max_length_tweet - len(word_embeddings)
            pad = [self.embeddings.index('<pad>')]
            word_embeddings.extend(pad * diff)
            word_embeddings_pad = word_embeddings

        return word_embeddings_pad

    def process_tweet(self, tweet):

        """
        Run all functions defined above
        """

        clean = self.clean_text(tweet)
        tokenized = self.tokenize_text(clean)
        word_embeddings = self.replace_token_with_index(tokenized)
        word_embeddings_pad = self.pad_sequence(word_embeddings)

        return word_embeddings_pad
