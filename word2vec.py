import re

import numpy as np


class Word2Vec:
    """
    Skip-Gram model implementation.
    Input: center word
    Output: predict context words
    """

    def __init__(
            self, 
            window_size: int = 5, 
            embedding_dim: int = 300, 
            learning_rate: float = 0.025,
            num_negative_samples: int = 5
    ):
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if num_negative_samples <= 0:
            raise ValueError("num_negative_samples must be > 0")

        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.num_negative_samples = num_negative_samples

        self.W_in = None # input matrix
        self.W_out = None # output matrix
        self.tokens = None  # Tokenized corpus
        self.vocab_dict = None


    @staticmethod
    def clean_text(corpus: str) -> list[str]:
        corpus = re.sub(r"[^a-zA-Z\s]", "", corpus) # Delete all non-letter characters
        corpus = corpus.casefold() # Convert all characters to lowercase
        words = corpus.split() # Split corpus into words
        return words
    

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid with protection from overflow"""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )


    def build_vocab(self, words: list[str]) -> None:
        self.vocab_dict = {}
        for word in words:
            if word not in self.vocab_dict:
                self.vocab_dict[word] = len(self.vocab_dict)


    def tokenize(self, corpus: list[str]) -> None:
        if self.vocab_dict is None:
            raise ValueError("vocab_dict is not built. Call build_vocab() first")
        self.tokens = np.array([self.vocab_dict[word] for word in corpus], dtype=np.uint32)


    def build_noise_distribution(self) -> None:
        """Unigram distribution raised to the 3/4 power"""
        counts = np.zeros(len(self.vocab_dict), dtype=np.float64)
        for idx in self.tokens:
            counts[idx] += 1
        counts = counts ** 0.75
        self.noise_dist = counts / counts.sum()


    def generate_training_pairs(self):
        if self.tokens is None:
            raise ValueError("Tokens is not built. Call tokenize() first.")
        for word in range(len(self.tokens)):
            for offset in range(-self.window_size, self.window_size + 1):
                if offset == 0: continue
                context = word + offset
                if 0 <= context < len(self.tokens):
                    yield np.array((self.tokens[word], self.tokens[context]))

    
    def softmax(self, vector: np.array) -> np.ndarray:
        return np.exp(vector) / np.sum(np.exp(vector))
    

    def sample_negatives(self, word: int) -> np.ndarray:
        """
        Draw n negative words from distribution,
        excluding the true context word to avoid false negatives.
        """
        vocab_size = len(self.vocab_dict)
        negatives = []
        while len(negatives) < self.num_negative_samples:
            candidates = np.random.choice(vocab_size, size=self.num_negative_samples, p=self.noise_dist)
            for context in candidates:
                if context != word:
                    negatives.append(context)
                if len(negatives) == self.num_negative_samples:
                    break
        return np.array(negatives, dtype=np.int32)
    

    def train_step(self) -> None:
        for word, context in self.generate_training_pairs():
            hidden = self.W_in[word]

            pos_score = hidden @ self.W_out[:, context]
            pos_sigmoid = self.sigmoid(pos_score)
            pos_error = pos_sigmoid - 1.0

            neg_idx = self.sample_negatives(context)
            neg_scores = hidden @ self.W_out[:, neg_idx]
            neg_sigmoid = self.sigmoid(neg_scores)
            neg_error = neg_sigmoid 

            d_hidden = (pos_error * self.W_out[:, context] + self.W_out[:, neg_idx] @ neg_error)

            self.W_out[:, context] -= self.learning_rate * pos_error * hidden
            self.W_out[:, neg_idx] -= self.learning_rate * np.outer(hidden, neg_error)

            self.W_in[word] -= self.learning_rate * d_hidden


    def evaluate(self) -> float:
        """Evaluate model performance"""
        total_loss = 0.0
        count = 0
        for word, context in self.generate_training_pairs():
            hidden = self.W_in[word]

            pos_score = hidden @ self.W_out[:, context]
            pos_loss  = -np.log(self.sigmoid(pos_score) + 1e-9)

            neg_idx    = self.sample_negatives(context)
            neg_scores = hidden @ self.W_out[:, neg_idx]
            neg_loss   = -np.sum(np.log(self.sigmoid(-neg_scores) + 1e-9))

            total_loss += pos_loss + neg_loss
            count += 1

        avg_loss = total_loss / count
        print(f"Average loss: {avg_loss:.4f}")
        return avg_loss


    def train(self, corpus: str, epochs: int) -> None:
        """Fit the model to the given data"""
        words = self.clean_text(corpus)
        self.build_vocab(words)
        self.tokenize(words)
        self.build_noise_distribution()

        self.W_in = np.random.uniform(-1, 1, (len(self.vocab_dict), self.embedding_dim))
        self.W_out = np.random.uniform(-1, 1, (self.embedding_dim, len(self.vocab_dict)))

        for epoch in range(epochs):
            print(f"Training epoch {epoch+1}/{epochs}")
            self.train_step()
            self.evaluate()