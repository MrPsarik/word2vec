import re

import numpy as np


FILE_PATH = "data\\AllCombined_half.txt"
TEXT_SIZE = 100_000


def text_load(file_path, size=-1) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read(size)
        return text 


def text_cleanup(text) -> list[str]:
    # Delete all non-letter characters 
    text = re.sub(r"[^a-zA-Z\s+]", "", text)

    # Convert all characters to lowercase
    text = text.casefold()

    # Split text into words
    text = text.split()

    return text


def text_to_dict(text) -> dict[str, int]:
    dict = {}
    text = set(text)

    for i, word in enumerate(text):
        dict[word] = i

    return dict


if __name__ == "__main__": 

    text = text_load(FILE_PATH, TEXT_SIZE)

    text = text_cleanup(text)

    dict = text_to_dict(text)