import pdb
import sys

import nltk
from nltk.tokenize import RegexpTokenizer


def load_data():

    with open("news_label.csv","r",encoding="utf-8") as f:
        raw_data=f.readlines()[1:]
        data=[]
        for each in raw_data:
            tmp=[]
            info_list=each.strip("\n").split(",")
            if len(info_list)==5:
                text=info_list[1].lower()
                tokenizer = RegexpTokenizer(r'\w+')
                text=tokenizer.tokenize(text)

                data.append([text,int(info_list[-1])])
            else:
                text=" ".join(info_list[1:-3])
                text = text.lower()
                tokenizer = RegexpTokenizer(r'\w+')
                text = tokenizer.tokenize(text)
                data.append([text, int(info_list[-1])])
    return data
