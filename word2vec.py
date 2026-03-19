import re

import numpy as np


class Word2Vec:
    """
    Skip-Gram model implementation.
    Input: center word
    Output: predict context words
    """

    def __init__(self, window_size: int = 5):
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.window_size = window_size
        self.tokens: np.ndarray[np.uint32] | None = None  # Tokenized corpus
        self.vocab_dict: dict[str, int] | None = None
        # self.vocabulary: np.ndarray | None = None


    @staticmethod
    def clean_text(corpus: str) -> list[str]:
        corpus = re.sub(r"[^a-zA-Z\s]", "", corpus) # Delete all non-letter characters
        corpus = corpus.casefold() # Convert all characters to lowercase
        words = corpus.split() # Split corpus into words
        return words


    # @staticmethod
    # def words_to_dict(words: list[str]) -> dict[str, int]:
    #     vocab_dict = {}
    #     for word in words:
    #         if word not in vocab_dict:
    #             vocab_dict[word] = len(vocab_dict)
    #     return vocab_dict
    

    # @staticmethod
    # def dict_to_vectors(vocab_dict: dict[str, int]) -> np.ndarray:
    #     vocab_size = len(vocab_dict)
    #     # Create a vocab_size x vocab_size identity matrix (one-hot vectors)
    #     return np.eye(vocab_size)


    def build_vocab(self, words: str) -> None:
        self.vocab_dict = {}
        for word in words:
            if word not in self.vocab_dict:
                self.vocab_dict[word] = len(self.vocab_dict)


    # def build_vocabulary(self) -> None:
    #     if self.vocab_dict is None:
    #         raise ValueError("vocab_dict is not built. Call build_dict() first.")
    #     self.vocabulary = self.dict_to_vectors(self.vocab_dict)


    def tokenize(self, corpus: str) -> None:
        if self.vocab_dict is None:
            raise ValueError("vocab_dict is not built. Call build_dict() first")
        self.tokens = np.array([self.vocab_dict[word] for word in corpus], dtype=np.uint32)


    def generate_training_pairs(self):
        for i in range(len(self.tokens)):
            left = max(i - self.window_size, 0)
            right = min(i + self.window_size, len(self.tokens))
            for token in self.tokens[left: right]:
                if self.tokens[i] == token: continue
                else: yield np.array((self.tokens[i], token))


    def train(self, corpus: str, epochs: int) -> None:
        """Fit the model to the given data"""
        words = self.clean_text(corpus)
        self.build_vocab(words)
        self.tokenize(words)

        for epoch in range(epochs):
            print(f"Training epoch {epoch+1}/{epochs}")
            # forward pass
            # negative sampling
            # loss
            # backward pass
        

    # Neural Network Architecture 
    # Input layer: one-hot vector of size V
    # Hidden layer: linear layer of size 300 (hyperparameter)
    # Output layer: softmax over V classes

    # Two weight matrices:
    # W (size V x N) input embedding matrix
    # W' (size N x V) output embedding matrix


    def predict(self):
        """Inference the model on the given input"""
        pass


    def evaluate(self):
        """Evaluate model performance"""
        pass
    