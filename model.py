import torch
from torch.nn import Module
from transformers import BertModel, BertTokenizer


class HumorDetectionModel(Module):
    def __init__(self, rnn_size, use_ambiguity=True, dropout=0.):
        super().__init__()
        self.use_ambiguity = use_ambiguity
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        word_embedding_size = 768  # size of BERT embedding
        if use_ambiguity:
            word_embedding_size += 1  # we will be enhancing the token embeddings with ambiguity scores
        self.rnn = torch.nn.LSTM(
            input_size=word_embedding_size,
            hidden_size=rnn_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.output_layer = torch.nn.Linear(rnn_size, out_features=2)

    def to(self, *args, **kwargs):
        self.bert = self.bert.to(*args, **kwargs)
        self.rnn = self.rnn.to(*args, **kwargs)
        self.output_layer = self.output_layer.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, inputs):
        token_indices = inputs["text"]
        ambiguity_scores = inputs["ambiguity"]
        ambiguity_scores = torch.unsqueeze(ambiguity_scores, dim=-1)
        attention_mask = inputs["pad_mask"]
        bert_embeds = self.bert(
            input_ids=token_indices,
            attention_mask=attention_mask
        ).last_hidden_state
        if self.use_ambiguity:
            bert_embeds = torch.cat((bert_embeds, ambiguity_scores), dim=-1)
        print(bert_embeds.size())
        rnn_output, (hn, cn) = self.rnn(bert_embeds)
        print(hn.size())
        hn = torch.sum(hn, dim=0)  # add the two directional layers
        print(hn.size())
        return self.output_layer(hn[:, -1, :])  # feed final LSTM output to the output layer


if __name__ == "__main__":
    model = HumorDetectionModel(rnn_size=5, dropout=.2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text = "This is a test."
    token_indices = tokenizer.encode(text, max_length=9, pad_to_max_length=True)
    pad_mask = torch.tensor([0 if index == 0 else 1 for index in token_indices], dtype=torch.float)
    print(tokenizer.decode(token_indices))
    print(pad_mask)
    token_indices = torch.tensor(token_indices, dtype=torch.long)
    ambiguity = torch.tensor([0, 0, 0, 0, 3, 0, 0, 0, 0], dtype=torch.float)
    input_dict = {
        "text": torch.unsqueeze(token_indices, dim=0),  # un-squeezing all inputs to add a batch dimension
        "ambiguity": torch.unsqueeze(ambiguity, dim=0),
        "pad_mask": torch.unsqueeze(pad_mask, dim=0)
    }
    model_output = model(input_dict)
    print(model_output)
    predicted_classes, class_scores = torch.max(model_output, dim=-1)
    print("Model predictions:", class_scores.tolist())
