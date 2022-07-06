import numpy as np
import torch


def generate_data(vals, num_examples=500):
    inpts = [np.random.randint(low=0, high=9, size=3)
             for _ in range(num_examples)]
    outputs = [ (inp[0] * np.sum(inp[1:])) for inp in inpts]
    tokens = [[vals[i] for i in inpt] for inpt in inpts]

    return tokens, outputs
