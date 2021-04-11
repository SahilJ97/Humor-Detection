import torch
from transformers import BertTokenizer
import csv
import os
import utils
import tokenizations
from tqdm import tqdm

# See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html for reference
class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, text_a, text_b, label, ambiguity, original):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.ambiguity = ambiguity
        self.original = original


class HumorDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, max_len, task='train', ambiguity_fn="wn"):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.label_map = {label: i for i, label in enumerate(["0", "1"])}
        self.max_len = max_len
        self.length = 0
        self.originals = []
        self.line_a = []
        self.line_b = []
        self.word_ambiguity = []
        self.labels = []

        if ambiguity_fn in ["wn", "none"]:
            extension = "_wordnet_amb.tsv"
        elif ambiguity_fn == "csi":
            extension = "_csi_amb.tsv"
        else:
            print("Error: ambiguity_fn must be one of ['none', 'wn', 'csi'].")
            exit(1)
        data_file = task + extension
        self.read_tsv(os.path.join(data_dir, data_file))

    def read_tsv(self, file_path):
        with open(file_path, "r", encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                label = line[1]

                text = line[3].split('_____')
                self.line_a.append(text[0])
                self.line_b.append(text[1])
                self.originals.append(line[3])

                self.labels.append(self.label_map[label])
                ambiguity = eval(line[4])
                self.word_ambiguity.append(ambiguity)

        self.length = len(self.line_a)

    def get_examples(self):
        examples = []

        # build examples
        for ix in range(self.length):
            ex = InputExample(
                text_a=self.line_a[ix],
                text_b=self.line_b[ix],
                label=self.labels[ix],
                ambiguity=self.word_ambiguity[ix],
                original=self.originals[ix]
            )
            examples.append(ex)
        return examples

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
    print('loading dataset')
    data_dir = 'data/'
    max_seq_len = 512
    task = 'train'
    train = HumorDetectionDataset(data_dir, max_seq_len, task)
    print('dataset loaded, length -', train.length)

    '''
    total, exceeded = 0, 0
    for item in tqdm(train):
        total += 1
        if len(item["text"]) > 512:
            exceeded += 1
    print(exceeded, total)
    '''

