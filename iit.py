import numpy as np
import torch
from random import shuffle

'''
def generate_data(vals, num_examples=500):
    inpts = [np.random.randint(low=0, high=9, size=3)
             for _ in range(num_examples)]
    outputs = [ (inp[0] * np.sum(inp[1:])) for inp in inpts]
    tokens = [[vals[i] for i in inpt] for inpt in inpts]

    return tokens, outputs
'''

def generate_data(vals, train_test_split=0.9):
    inpts = list()
    for i in range(10):
        for j in range(10):
            for k in range(10):
                seq = [i, j, k]
                if seq not in inpts:
                    inpts.append(seq)
    shuffle(inpts)
    outputs = [ (inp[0] * np.sum(inp[1:])) for inp in inpts]
    tokens = [[vals[i] for i in inpt] for inpt in inpts]
    split = int(len(inpts) * train_test_split)
    return tokens[:split], outputs[:split], tokens[split:], outputs[split:]
