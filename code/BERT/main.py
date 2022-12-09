import csv
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset,DataLoader
from transformers import BertTokenizer
from random import shuffle
from utils import load_data
import sys
from sklearn.metrics import classification_report
data=load_data()
shuffle(data)
train_data=data[0:int(len(data)*0.8)]
test_data=data[int(len(data)*0.8):]

tokenized_train,tokenized_test=[],[]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class NLIDATASET(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data=data
        self.tokenized_train=[]
        for each in self.data:
            text, label = each[0],each[1]
            text_encoded_text = tokenizer.encode_plus(
                (" ").join(text), add_special_tokens=True, truncation=True,
                max_length=64, padding='max_length',
                return_attention_mask=True,
                return_tensors='pt')
            self.tokenized_train.append([text_encoded_text, label])
    def __getitem__(self, index):
        text=self.tokenized_train[index][0]['input_ids'][0]
        text_mask=self.tokenized_train[index][0]['attention_mask'][0]
        label=int(self.tokenized_train[index][1])
        
        return text,text_mask,label

    def __len__(self):
        return len(self.data)


train_dataset=NLIDATASET(train_data)
test_dataset=NLIDATASET(test_data)
print("----------Finish Loading Data---------")
batch_size=8
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle = True)
dev_loader = torch.utils.data.DataLoader(test_dataset,batch_size = batch_size)
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from model import NLIModel
net = NLIModel()

print("----------Finish Loading Model---------")
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-5)
trainloss=[]
devacc=[]
epochs = 70
save_path = './bestModelbase.pth'
best_acc = 0.0
train_steps = len(train_loader)
for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        text,text_mask, labels = data
        #print(labels)
        optimizer.zero_grad()
        outputs = net(text.to(device),text_mask.to(device))

        loss = loss_function(outputs.logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    # dev
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        dev_bar = tqdm(dev_loader)
        label_true=[]
        label_pre=[]
        for test_data in dev_bar:
            dev_text,dev_text_mask, dev_labels = test_data
            outputs = net(dev_text.to(device),dev_text_mask.to(device))
            predict_y = torch.max(outputs.logits, dim=1)[1]
            acc += torch.eq(predict_y, dev_labels.to(device)).sum().item()
            label_true.extend(dev_labels.numpy().tolist())
            label_pre.extend(predict_y.cpu().numpy().tolist())

    dev_accurate = acc / (len(dev_loader)*batch_size)
    print('[epoch %d] train_loss: %.3f  dev_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, dev_accurate))
    print(classification_report(label_true,label_pre))
    trainloss.append(running_loss / train_steps)
    devacc.append(dev_accurate)

    if dev_accurate >= best_acc:
        best_acc = dev_accurate
        torch.save(net.state_dict(), save_path)
print(trainloss)
print(devacc)
print(best_acc)
print('Finished Training')