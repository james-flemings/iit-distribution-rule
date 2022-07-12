import numpy as np
import torch
import copy
from random import shuffle

'''
def generate_data(vals, num_examples=500):
    inpts = [np.random.randint(low=0, high=9, size=3)
             for _ in range(num_examples)]
    outputs = [ (inp[0] * np.sum(inp[1:])) for inp in inpts]
    tokens = [[vals[i] for i in inpt] for inpt in inpts]

    return tokens, outputs
'''

def generate_data(vals: list[str], train_test_split:float = 0.9):
    inpts = list()
    for i in range(len(vals)):
        for j in range(len(vals)):
            for k in range(len(vals)):
                seq = [i, j, k]
                if seq not in inpts:
                    inpts.append(seq)
    shuffle(inpts)
    outputs = [ (inp[0] * np.sum(inp[1:])) for inp in inpts]
    tokens = [[vals[i] for i in inpt] for inpt in inpts]
    split = int(len(inpts) * train_test_split)
    return tokens[:split], outputs[:split], tokens[split:], outputs[split:]


def get_iit_distribution_dataset_both(variable: int, vals: list[str], train_test_split: float = 0.9, first: bool = False):
    x_base_train, y_base_train, x_base_test, y_base_test = generate_data(vals, train_test_split)

    x_source_train = copy.deepcopy(x_base_train)
    x_source_test = copy.deepcopy(x_base_test)

    shuffle(x_source_train)
    shuffle(x_source_test)

    y_source_train = get_iit_label(variable, vals, x_base_train, x_source_train, first)
    y_source_test = get_iit_label(variable, vals, x_base_test, x_source_test, first)

    return (x_base_train, y_base_train, x_source_train, y_source_train), (x_base_test, y_base_test, x_source_test, y_source_test)


def get_iit_label(variable: int, vals: list[str], x_base: list[list[str]], x_source: list[list[str]], first: bool):
    y_iit = []
    for base, source in zip(x_base, x_source):
        a = vals.index(source[0])
        b = vals.index(source[1])
        c = vals.index(source[2])
        x = vals.index(base[0])
        y = vals.index(base[1])
        z = vals.index(base[2])

        if first:
            sum = x * (b + c)
        else:
            sum = x*y + a*c if variable == 2 else a*b + x*z
        y_iit.append(sum)

    return y_iit



vals = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
train_data, test_dat = get_iit_distribution_dataset_both(1, vals)
