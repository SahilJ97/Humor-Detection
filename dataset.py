import torch
from transformers import BertTokenizer
import csv
import utils
import tokenizations

# See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html for reference


class HumorDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, max_len):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.label_map = {label: i for i, label in enumerate(["0", "1"])}
        self.max_len = max_len
        self.length = 0
        self.lines = []
        self.word_ambiguity = []
        self.labels = []
        self.read_tsv(file_path)

    def read_tsv(self, file_path):
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                text_a = utils.prepare_text(line[3])
                label = line[1]
                ambiguity = eval(line[4])
                self.lines.append(text_a)
                self.word_ambiguity.append(ambiguity)
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
            truncation=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        attn_mask = torch.tensor(encoded['attention_mask'], dtype=torch.float)

        # Align ambiguity scores with BERT tokens
        bert_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        alt_tokens, ambiguity_scores = zip(*self.word_ambiguity[index])
        bert_to_alt, alt_to_bert = tokenizations.get_alignments(bert_tokens, alt_tokens)
        aligned_scores = [0]*self.max_len
        for i in range(len(alt_to_bert)):
            for j in alt_to_bert[i]:
                aligned_scores[j] = ambiguity_scores[i]

        return {
            "text": input_ids,
            "ambiguity": torch.tensor(aligned_scores),
            "pad_mask": attn_mask,
            "label": torch.tensor(label)
        }


if __name__ == "__main__":
    train = HumorDetectionDataset('data/dev_with_amb.tsv', 512)
    print(train.length)
    total, exceeded = 0, 0
    for item in train:
        total += 1
        if len(item["text"]) > 512:
            exceeded += 1
    print(exceeded/total)

