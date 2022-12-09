import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import BertForSequenceClassification, BertConfig
class NLIModel(torch.nn.Module):
    def __init__(self):
        super(NLIModel, self).__init__()
        config = BertConfig.from_pretrained("bert-base-uncased",
                                            num_labels=6,
                                            hidden_dropout_prob=0.2, 
                                            output_attentions = False,
                                            output_hidden_states = False)
        self.bert =BertForSequenceClassification.from_pretrained('bert-base-uncased',config=config)
    def forward(self,text,text_mask):
        x = self.bert(text, text_mask)
        return x

