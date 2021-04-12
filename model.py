import torch
from torch.nn import Module, CrossEntropyLoss
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
        self.loss_fnc = CrossEntropyLoss()
        self.output_layer = torch.nn.Linear(rnn_size*2, out_features=2)

    def to(self, *args, **kwargs):
        self.bert = self.bert.to(*args, **kwargs)
        self.rnn = self.rnn.to(*args, **kwargs)
        self.output_layer = self.output_layer.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, input_ids, attention_mask, token_type_ids, ambiguity_scores, labels=None):
        ambiguity_scores = torch.unsqueeze(ambiguity_scores, dim=-1)
        bert_embeds = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        if self.use_ambiguity:
            bert_embeds = torch.cat((bert_embeds, ambiguity_scores), dim=-1)
        rnn_output, (hn, cn) = self.rnn(bert_embeds)
        logits = self.output_layer(rnn_output[:, -1, :])  # feed final LSTM output to the output layer

        ## transformers expects output tuple (loss, )
        output = ()
        if labels is not None:
            loss = self.loss_fnc(logits, labels)
            output = (loss,)

        output += (logits,)

        return output


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
        "token_indices": torch.unsqueeze(token_indices, dim=0),  # un-squeezing all inputs to add a batch dimension
        "attention_mask": torch.unsqueeze(pad_mask, dim=0),
        "ambiguity_scores": torch.unsqueeze(ambiguity, dim=0)
    }
    model_output = model(**input_dict)
    print(model_output)

    # Example with labels
    label = torch.tensor([1])
    input_dict['labels'] = label
    model_output = model(**input_dict)
    print('Loss:', model_output[0])

    # Getting predictions - return is (loss, logits) if labels are passed
    class_scores = torch.argmax(model_output[1], dim=-1)
    print("Model predictions:", class_scores.tolist())
