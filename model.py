# hacky solution to avoid adding __init__.py in cs224u
from pyrsistent import freeze
import torch.utils.data
import torch.nn as nn
import torch
import numpy as np
import sys

sys.path.insert(0, './cs224u')

from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

class ActivationLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_activation):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, device=device)
        self.activation = hidden_activation

    def forward(self, x):
        return self.activation(self.linear(x))


class TorchDeepNeuralModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 output_size,
                 num_inputs,
                 device,
                 hidden_activation,
                 num_layers=1,
                 embed_dim=40,
                 hidden_dim=50,
                 embedding=None,
                 freeze_embedding=False):

        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim

        self.input_size = self.num_inputs * self.embed_dim
        self.output_size = output_size
        self.device = device
        self.hidden_activation = hidden_activation

        # define computation graph
        # Input to embedding
        self.embedding = self._define_embedding(
            embedding, self.vocab_size, self.embed_dim, freeze_embedding)
        # Embedding to hidden:
        self.layers = [
            ActivationLayer(
                self.input_size, self.hidden_dim, self.device, self.hidden_activation
            )
        ]
        # Hidden to hidden
        for _ in range(self.num_layers-1):
            self.layers += [
                ActivationLayer(
                    self.hidden_dim, self.hidden_dim, self.device, self.hidden_activation
                )
            ]
        self.layers.append(
            nn.Linear(self.hidden_dim, self.output_size, device=self.device)
        )
        self.model = nn.Sequential(*self.layers)

    def forward(self, X):
        X = self.embedding(X)
        new_x = []
        for x in X:
            new_x.append(torch.cat(tuple(x[i]
                         for i in range(self.num_inputs))))
        new_x = torch.stack(new_x)
        output = self.model(new_x)
        return output

    @staticmethod
    def _define_embedding(embedding, vocab_size, embed_dim, freeze_embedding):
        if embedding is None:
            emb = nn.Embedding(vocab_size, embed_dim)
            emb.weight.requires_grad = not freeze_embedding
            return emb
        elif isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
            return nn.Embedding.from_pretrained(
                embedding, freeze=freeze_embedding
            )
        else:
            return embedding


class TorchDeepNeuralClassifier(TorchShallowNeuralClassifier):
    def __init__(self,
                 vocab,
                 output_size,
                 num_inputs,
                 num_layers=1,
                 embed_dim=40,
                 embedding=None, 
                 freeze_embedding=False,
                 **base_kwargs):
        self.num_layers = num_layers
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.embed_dim = embed_dim
        self.num_inputs = num_inputs

        super().__init__(**base_kwargs)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.params += ['num_layers']
        self.output_size = output_size

        self.embedding = embedding
        self.freeze_embedding = freeze_embedding


    def build_dataset(self, X, y=None):
        new_X = []
        index = dict(zip(self.vocab, range(self.vocab_size)))
        for ex in X:
            seq = [index[w] for w in ex]
            seq = torch.tensor(seq)
            new_X.append(seq)

        X = torch.stack(new_X)

        if y is None:
            dataset = torch.utils.data.TensorDataset(X)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(X, y)
        return dataset

    def build_graph(self):
        """
        Define the model's computation graph.

        Returns
        -------
        nn.Module

        """
        return TorchDeepNeuralModel(self.vocab_size, self.output_size,
                                    self.num_inputs, self.device, self.hidden_activation,
                                    self.num_layers, self.embed_dim, self.hidden_dim,
                                    self.embedding, self.freeze_embedding)
