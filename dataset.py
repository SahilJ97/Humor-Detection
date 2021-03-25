import torch
from transformers import BertTokenizer

# See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html for reference


class HumorDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, max_len):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # TODO: load data (loop through data, encoding it with self.tokenizer and saving data/labels using
        #  class variables, etc.)
        return

    def __len__(self):
        # TODO: return length of dataset
        return 0

    def __getitem__(self, index):
        # TODO: pad/crop text encoding sequence, return dict of PyTorch tensors (no batch dimension)
        return {
            "text": None,
            "ambiguity": None,
            "pad_mask": None
        }
