from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup


class BertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels, dropout=0.2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        pooled_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def get_model(model_name, num_labels):
    model = BertClassifier(model_name, num_labels)
    return model
