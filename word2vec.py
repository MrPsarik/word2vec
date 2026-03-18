import re

import numpy as np


class Word2Vec:
    """
    Skip-Gram model implementation.
    Input: center word
    Output: predict context words
    Context window size: 10
    """

    def __init__(self):
        self.vocab_dict = None
        self.vocabulary = None


    @staticmethod
    def clean_text(corpus: str) -> list[str]:
        # Delete all non-letter characters 
        corpus = re.sub(r"[^a-zA-Z\s]", "", corpus)

        # Convert all characters to lowercase
        corpus = corpus.casefold()

        # Split corpus into words
        corpus = corpus.split()

        return corpus


    @staticmethod
    def words_to_dict(words: list[str]) -> dict[str, int]:
        vocab_dict = {}

        for word in words:
            if word not in vocab_dict:
                vocab_dict[word] = len(vocab_dict)
        return vocab_dict
    

    @staticmethod
    def dict_to_vectors(vocab_dict: dict[str, int]) -> np.ndarray:
        vocab_size = len(vocab_dict)
        # Create a vocab_size x vocab_size identity matrix (one-hot vectors)
        return np.eye(vocab_size)


    def build_dict(self, corpus: str) -> None:
        words = self.clean_text(corpus)
        self.vocab_dict = self.words_to_dict(words)


    # Given a corpus, build a vocabulary of one-hot vectors
    def build_vocabulary(self) -> None:
        if self.vocab_dict is None:
            raise ValueError("vocab_dict is not built. Call build_dict() first.")
        self.vocabulary = self.dict_to_vectors(self.vocab_dict)


    # Neural Network Architecture 
    # Input layer: one-hot vector of size V
    # Hidden layer: linear layer of size 300 (hyperparameter)
    # Output layer: softmax over V classes

    # Two weight matrices:
    # W (size V x N) input embedding matrix
    # W' (size N x V) output embedding matrix