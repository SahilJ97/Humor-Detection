import torch
from transformers import BertTokenizer
import csv
import utils

# See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html for reference


class HumorDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, max_len):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.label_map = {label: i for i, label in enumerate(["0", "1"])}
        self.max_len = max_len
        self.length = 0
        self.lines = []
        self.labels = []
        self.read_tsv(file_path)

    def read_tsv(self, file_path):
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                text_a = utils.prepare_text(line[3])
                label = line[1]
                self.lines.append(text_a)
                self.labels.append(self.label_map[label])
        self.length = len(self.lines)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        text = self.lines[index]
        label = self.labels[index]
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_special_tokens_mask=True
        )

        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        attn_mask = torch.tensor(encoded['attention_mask'], dtype=torch.float)

        return {
            "text": input_ids,
            "ambiguity": torch.tensor([0]*self.max_len, dtype=torch.float),  # FIXME
            "pad_mask": attn_mask,
            "label": torch.tensor(label)
        }


if __name__ == "__main__":
    train = HumorDetectionDataset('data/dev.tsv', 128)
    print(train.length)
    print(train[0])
