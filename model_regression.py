import torch
import torch.nn as nn
from model_classifier import ActivationLayer

from torch_model_base import TorchModelBase

class TorchLinearRegressionModel(nn.Module):
    def __init__(self, 
                vocab_size,
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

        self.w = nn.Parameter(torch.zeros(self.hidden_dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.model = nn.Sequential(*self.layers)

    def forward(self, X):
        X = torch.tensor(X).to(self.device).long()
        X = self.embedding(X)

        new_x = []
        for x in X:
            new_x.append(torch.cat(tuple(x[i]
                         for i in range(self.num_inputs))))
        new_x = torch.stack(new_x)

        h = self.model(new_x)
        return h.matmul(self.w) + self.b

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
        elif isinstance(embedding, pd.DataFrame): 
            embedding = torch.FloatTensor(embedding.to_numpy())
            return nn.Embedding.from_pretrained(embedding, freeze=freeze_embedding)
        else:
            return embedding


class TorchLinearRegression(TorchModelBase):
    def __init__(self, **base_kwargs):
        super().__init__(**base_kwargs)
        self.loss = nn.MSELoss(reduction="mean")

    def build_graph(self):
        return TorchLinearRegressionModel(self.input_dim)

    def build_dataset(self, X, y=None):
        """
        This function will be used in training (when there is a `y`)
        and in prediction (no `y`). For both cases, we rely on a
        `TensorDataset`.
        """
        X = torch.FloatTensor(X)
        self.input_dim = X.shape[1]
        if y is None:
            dataset = torch.utils.data.TensorDataset(X)
        else:
            y = torch.FloatTensor(y)
            dataset = torch.utils.data.TensorDataset(X, y)
        return dataset

    def predict(self, X, device=None):
        """
        The `_predict` function of the base class handles all the
        details around data formatting. In this case, the
        raw output of `self.model`, as given by
        `TorchLinearRegressionModel.forward` is all we need.
        """
        return self._predict(X, device=device).cpu().numpy()

    def score(self, X, y):
        """
        Follow sklearn in using `r2_score` as the default scorer.
        """
        preds = self.predict(X)
        return r2_score(y, preds)