# Tokenizer.py

import re
from Loader import WebLoader, MWELoader
from Token import Token


def Tokenizer(text, web_loader, mwe_loader):
    # First pass: Split by spaces
    tokens = re.findall(r"\S+", text)

    # Second pass: Separate punctuation from the ends of words
    intermediate_tokens = []
    for token in tokens:
        if not token[-1].isalnum():
            intermediate_tokens.append(token[:-1])
            intermediate_tokens.append(token[-1])
        else:
            intermediate_tokens.append(token)
    tokens = intermediate_tokens

    # Third pass: Handle special cases for apostrophes
    intermediate_tokens = []
    for token in tokens:
        if "'" in token[:3]:
            # If the token includes an apostrophe at the beginning, split it into two tokens
            index = token.find("'")
            intermediate_tokens.append(token[: index + 1])
            intermediate_tokens.append(token[index + 1 :])
        else:
            # Otherwise, just add the token as is
            intermediate_tokens.append(token)
    tokens = intermediate_tokens

    # Fourth pass, handle `-`
    intermediate_tokens = []
    for token in tokens:
        if "-" in token and len(Token(token, web_loader).lemmas) != 0:
            # If the token includes a hyphen and is a valid word, don't split it
            intermediate_tokens.append(token)
        else:
            parts = token.split("-")
            intermediate_tokens.extend(parts)
    tokens = intermediate_tokens

    # Create Token objects for each tokenized word
    token_objects = []
    for token in tokens:
        if token.strip():
            token_objects.append(Token(token, web_loader))

    return token_objects


if __name__ == "__main__":
    text = "auriez-vous la gentiesse de me dire pourquoi."
    web_loader = WebLoader("")
    mwe_loader = None  # MWELoader("data/mwe.txt")
    tokens = Tokenizer(text, web_loader, mwe_loader)
    for token in tokens:
        print(token)
