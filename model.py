import torch
from torch.nn import Module
from transformers import BertModel


class HumorDetectionModel(Module):
    def __init__(self, use_ambiguity=True):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # TODO: initialize model components

    def to(self, *args, **kwargs):
        # TODO: "self.param = self.param.to(*args, **kwargs)" for all "param"
        return super().to(*args, **kwargs)

    def forward(self, inputs):
        # TODO: implement forward pass
        return torch.zeros(1)
