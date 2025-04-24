# model.py
import torch.nn as nn
from transformers import BertModel
import torch

class ConcisenessClassifier(nn.Module):
    def __init__(self):
        super(ConcisenessClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, 3)  # 3 classes: 3_words, 5_words, 7_words

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        output = self.fc(pooled_output)
        return output
