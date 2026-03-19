from word2vec import Word2Vec


FILE_PATH = "data\\AllCombined_half.txt"
TEXT_SIZE = 100_000


def text_load(file_path, size=-1) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read(size)
        return text 


if __name__ == "__main__": 

    corpus = text_load(FILE_PATH, TEXT_SIZE)

    model = Word2Vec()

    model.train(corpus, 5)