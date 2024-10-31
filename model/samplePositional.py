import numpy as np
import math

def tokenize_string(input_string):
    """
    Tokenizes the input string by splitting it on spaces and returns the tokens.
    """
    return input_string.split()

def get_positional_encoding(token_length, d_model):
    """
    Computes positional encoding for each token position in the input sequence.
    
    Args:
    token_length: The length of the sequence (number of tokens).
    d_model: The dimension of the embedding space (model size).
    
    Returns:
    A matrix of shape (token_length, d_model) containing the positional encodings.
    """
    # Initialize with float32 dtype
    positional_encoding = np.zeros((token_length, d_model), dtype=np.float32)
    
    for pos in range(token_length):
        for i in range(0, d_model, 2):
            positional_encoding[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
    
    return positional_encoding

def getPos(input_string, d_model):
    # Tokenize the string
    tokens = tokenize_string(input_string)
    print("Tokens:", tokens)
    
    # Get the positional encoding for the tokens
    positional_encoding = get_positional_encoding(len(tokens), d_model)
    
    print("Positional Encoding:")
    print(positional_encoding)


