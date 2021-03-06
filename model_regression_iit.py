from collections import defaultdict
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from model_regression import TorchLinearEmbeddingRegression

__author__ = "Atticus Geiger"
__version__ = "CS224u, Stanford, Spring 2022"


class IITModel(torch.nn.Module):
    def __init__(self, model, layers, id_to_coords,device):
        super().__init__()
        self.model = model
        self.layers = layers

        self.id_to_coords = defaultdict(lambda: defaultdict(list))
        for k, vals in id_to_coords.items():
            for d in vals:
                layer = d['layer']
                self.id_to_coords[k][layer].append(d)

        self.device = device

    def no_IIT_forward(self, X):
        return self.model(X)

    def forward(self, X):
        base = X[:,0,:].squeeze(1).type(torch.LongTensor).to(self.device)
        coord_ids = X[:,1,:].squeeze(1).type(torch.LongTensor).to(self.device)
        sources = X[:,2:,:].to(self.device)
        sources = [sources[:,j,:].squeeze(1).type(torch.LongTensor).to(self.device)
                   for j in range(sources.shape[1])]
        gets = self.id_to_coords[int(coord_ids.flatten()[0])]
        sets = copy.deepcopy(gets)
        self.activation = dict()

        for layer in gets:
            for i, get in enumerate(gets[layer]):
                handlers = self._gets_sets(gets ={layer: [get]},sets = None)
                source_logits = self.no_IIT_forward(sources[i])
                for handler in handlers:
                    handler.remove()
                sets[layer][i]["intervention"] = self.activation[f'{get["layer"]}-{get["start"]}-{get["end"]}']

        base_logits = self.no_IIT_forward(base)
        handlers = self._gets_sets(gets = None, sets = sets)
        counterfactual_logits = self.no_IIT_forward(base)
        for handler in handlers:
            handler.remove()

        return counterfactual_logits, base_logits

    def make_hook(self, gets, sets, layer):
        def hook(model, input, output):
            layer_gets, layer_sets = [], []
            if gets is not None and layer in gets:
                layer_gets = gets[layer]
            if sets is not None and layer in sets:
                layer_sets = sets[layer]
            for set in layer_sets:
                output = torch.cat([output[:,:set["start"]], set["intervention"], output[:,set["end"]:]], dim = 1)
            for get in layer_gets:
                self.activation[f'{get["layer"]}-{get["start"]}-{get["end"]}'] = output[:,get["start"]: get["end"] ]
            return output
        return hook

    def _gets_sets(self,gets=None, sets = None):
        handlers = []
        for layer in range(len(self.layers)):
            hook = self.make_hook(gets,sets, layer)
            both_handler = self.layers[layer].register_forward_hook(hook)
            handlers.append(both_handler)
        return handlers

    def retrieve_activations(self, input, get, sets):
        input = input.type(torch.FloatTensor).to(self.device)
        self.activation = dict()
        get_val = {get["layer"]: [get]} if get is not None else None
        set_val = {sets["layer"]: [sets]} if sets is not None else None
        handlers = self._gets_sets(get_val, set_val)
        logits = self.model(input)
        for handler in handlers:
            handler.remove()
        return self.activation[f'{get["layer"]}-{get["start"]}-{get["end"]}']


class MSELossIIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="mean")

    def forward(self, preds, labels):
        return self.loss(preds[0], labels[: , 0]) + self.loss(preds[1], labels[:,1])


class TorchLinearEmbeddingRegressionIIT(TorchLinearEmbeddingRegression):
    def __init__(self, id_to_coords=None, **base_kwargs):
        super().__init__(**base_kwargs)
        self.loss = MSELossIIT()
        self.id_to_coords = id_to_coords
        self.shuffle_train = False

    def build_graph(self):
        model = super().build_graph()
        IITmodel = IITModel(model, model.layers, self.id_to_coords, self.device)
        return IITmodel

    def batched_indices(self, max_len):
        batch_indices = [x for x in range((max_len // self.batch_size))]
        output = []
        while len(batch_indices) != 0:
            batch_index = random.sample(batch_indices, 1)[0]
            batch_indices.remove(batch_index)
            output.append([batch_index*self.batch_size + x for x in range(self.batch_size)])
        return output

    def build_dataset(self, base, sources, base_y, IIT_y, coord_ids):
        new_base = []
        index_base = dict(zip(self.vocab, range(self.vocab_size)))
        for ex in base:
            seq = [index_base[w] for w in ex]
            seq = torch.LongTensor(seq)
            new_base.append(seq)
        base = torch.stack(new_base)

        new_sources = []
        index_sources = dict(zip(self.vocab, range(self.vocab_size)))
        for ex in sources[0]:
            seq = [index_sources[w] for w in ex]
            seq = torch.LongTensor(seq)
            new_sources.append(seq)
        sources = [torch.stack(new_sources)]

        self.input_dim = base.shape[1]
        coord_ids = torch.LongTensor(np.array(coord_ids))

        if base_y is None:
            return torch.stack([base, coord_ids.unsqueeze(1).expand(-1, base.shape[1])] + sources, dim=1)

        base_y = torch.FloatTensor(base_y)
        IIT_y = torch.FloatTensor(IIT_y)

        bigX = torch.stack([base, coord_ids.unsqueeze(1).expand(-1, base.shape[1])] + sources, dim=1)
        bigy = torch.stack((IIT_y, base_y), dim=1)
        dataset = torch.utils.data.TensorDataset(bigX, bigy)
        return dataset

    def prep_input(self, base, sources, coord_ids):
        bigX = torch.stack([base, coord_ids.unsqueeze(1).expand(-1, base.shape[1])] + sources, dim=1)
        return bigX

    def iit_predict(self, base, sources, coord_ids):
        #IIT_test = self.prep_input(base, sources, coord_ids)
        IIT_test = self.build_dataset(base, sources, None, None, coord_ids)
        IIT_preds, base_preds = self.model(IIT_test)
        return torch.round(IIT_preds), torch.round(base_preds)


if __name__ == '__main__':
    import iit
    from sklearn.metrics import classification_report
    import utils

    utils.fix_random_seeds()

    V1 = 0
    data_size = 10000
    embedding_dim = 4

    id_to_coords = {
        V1: [{"layer": 1, "start": 0, "end": embedding_dim}]
    }

    iit_equality_dataset = iit.get_IIT_equality_dataset(
        "V1", embedding_dim, data_size)

    X_base_train, X_sources_train, y_base_train, y_IIT_train, interventions = iit_equality_dataset

    model = TorchLinearEmbeddingRegressionIIT(
        hidden_dim=embedding_dim*4,
        hidden_activation=torch.nn.ReLU(),
        num_layers=3,
        id_to_coords=id_to_coords)

    model.fit(
        X_base_train,
        X_sources_train,
        y_base_train,
        y_IIT_train,
        interventions)

    X_base_test, X_sources_test, y_base_test, y_IIT_test, interventions = iit.get_IIT_equality_dataset(
        "V1", embedding_dim, 100)

    IIT_preds, base_preds = model.iit_predict(
        X_base_test, X_sources_test, interventions)

    print("\nStandard evaluation")
    print(classification_report(y_base_test, base_preds))

    print("V1 counterfactual evaluation")
    print(classification_report(y_IIT_test, IIT_preds))
