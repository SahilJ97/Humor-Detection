import torch
from torch.nn import Module
from transformers import BertModel


class HumorDetectionModel(Module):
    def __init__(self, seq_len, rnn_size, use_ambiguity=True, dropout=0.):
        super().__init__()
        self.use_ambiguity = use_ambiguity
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        word_embedding_size = 768  # size of BERT token embedding
        if use_ambiguity:
            word_embedding_size += 1  # we will be enhancing the token embeddings with ambiguity scores
        self.rnn = torch.nn.LSTM(
            input_size=(seq_len, word_embedding_size),
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
        bert_embeds, _ = self.bert_model.forward(
            input_ids=token_indices,
            attention_mask=attention_mask
        )
        if self.use_ambiguity:
            bert_embeds = torch.cat((bert_embeds, ambiguity_scores), dim=-1)
        rnn_output = self.rnn(bert_embeds)
        return self.output_layer(rnn_output)
