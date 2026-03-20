import re

import numpy as np


class Word2Vec:
    """
    Skip-Gram model implementation.
    Input: center word
    Output: predict context words
    """

    def __init__(self, window_size: int = 5, embedding_dim: int = 300, learning_rate: float = 0.025):
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.window_size = window_size
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
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


    def build_vocab(self, words: list[str]) -> None:
        self.vocab_dict = {}
        for word in words:
            if word not in self.vocab_dict:
                self.vocab_dict[word] = len(self.vocab_dict)


    def tokenize(self, corpus: list[str]) -> None:
        if self.vocab_dict is None:
            raise ValueError("vocab_dict is not built. Call build_vocab() first")
        self.tokens = np.array([self.vocab_dict[word] for word in corpus], dtype=np.uint32)


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
    

    def train_step(self) -> None:
        for word, context in self.generate_training_pairs():
            hidden = self.W_in[word]
            logits = hidden @ self.W_out
            prob = self.softmax(logits)

            target = np.zeros(len(self.vocab_dict))
            target[context] = 1

            error = prob - target

            dW_out = np.outer(hidden, error)
            d_hidden = self.W_out @ error 

            self.W_out -= self.learning_rate * dW_out
            self.W_in[word] -= self.learning_rate * d_hidden


    def train(self, corpus: str, epochs: int) -> None:
        """Fit the model to the given data"""
        words = self.clean_text(corpus)
        self.build_vocab(words)
        self.tokenize(words)

        self.W_in = np.random.uniform(-1, 1, (len(self.vocab_dict), self.embedding_dim))
        self.W_out = np.random.uniform(-1, 1, (self.embedding_dim, len(self.vocab_dict)))

        for epoch in range(epochs):
            print(f"Training epoch {epoch+1}/{epochs}")
            self.train_step()
            self.evaluate()


    def predict(self):
        """Inference the model on the given input"""
        pass


    def evaluate(self) -> float:
        """Evaluate model performance"""
        total_loss = 0.0
        count = 0
        for word, context in self.generate_training_pairs():
            hidden = self.W_in[word]
            logits = hidden @ self.W_out
            prob = self.softmax(logits)

            # cross-entropy loss: -log(prob of true context word)
            total_loss += -np.log(prob[context] + 1e-9)   # 1e-9 avoids log(0)
            count += 1

        avg_loss = total_loss / count
        print(f"Average loss: {avg_loss:.4f}")
        return avg_loss