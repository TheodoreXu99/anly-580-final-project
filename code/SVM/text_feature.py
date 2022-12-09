import csv
import pdb
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer,BertModel,BertPreTrainedModel
import sys
import random
import jieba
from utils import load_data
text_data=load_data()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

import os
from  model import NLIModel

model=NLIModel()
weights_path = "../bert/bestModelbase.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()
text_features=[]
for each in text_data:
    text=" ".join(each[0])
    text_encoded_text = tokenizer.encode_plus(
        text, add_special_tokens=True, truncation=True,
        max_length=64, padding='max_length',
        return_attention_mask=True,
        return_tensors='pt')
    #pdb.set_trace()
    text = text_encoded_text['input_ids'][0].unsqueeze(0).to(device)
    text_mask = text_encoded_text['attention_mask'][0].unsqueeze(0).to(device)
    with torch.no_grad():
        output=model.bert.bert(text,text_mask)
        text_feature=output["pooler_output"].cpu()
        text_features.append((each[1],text_feature))

torch.save(text_features, 'text_features.pt')
pdb.set_trace()

