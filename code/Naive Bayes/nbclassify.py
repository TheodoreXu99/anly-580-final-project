import os
import json
import glob
import pdb
import sys
import string
from utils import load_data
from random import shuffle

data=load_data()
shuffle(data)

test_data=data[int(len(data)*0.8):]
data=data[:int(len(data)*0.8)]
six_classes = []
six_classes = [i[1] for i in data]

vocab_set = set()
all_files_list = glob.glob('op_spam_training_data/*/*/*/*.txt')

def get_dict():
    words_dict={}
    class_word_dict = {}
    for i in range(len(data)):
        words = data[i][0]
        #print(words)
        class_ref = six_classes[i]
        wordVsCount = class_word_dict.get(class_ref, {})
        for word in words:
            sum_word_count=words_dict.get(word,0)
            words_dict[word]=sum_word_count+1
            wordcount = wordVsCount.get(word, 0)
            wordVsCount[word] = wordcount + 1
            class_word_dict[class_ref] = wordVsCount
    return words_dict,class_word_dict

def remove_low_freq_words(words_dict,freq):
    new_word_dict = []
    sorted_word_list = sorted(words_dict.items(), key=lambda dic: dic[1])
    for word, word_count in sorted_word_list:
        if word_count >= freq:
            new_word_dict.append(word)
    return new_word_dict

class_count={0:six_classes.count(0),1:six_classes.count(1),2:six_classes.count(2),3:six_classes.count(3),4:six_classes.count(4),5:six_classes.count(5)}

whole_words_dict,each_class_word_dict=get_dict()

for class_,Words_Count in each_class_word_dict.items():
    currCount = 0
    for Words,Count in Words_Count.items():
        currCount += Count
        vocab_set.add(Words)
    class_count[class_] = currCount
vocal_set_refresh=remove_low_freq_words(whole_words_dict,2)


lambda_a = 0.6 #拉普拉斯平滑

each_class_word_probablities={}

for class_,word_count in each_class_word_dict.items():
        wordVsProb = each_class_word_probablities.get(class_,{})
        for vocabword in vocal_set_refresh:
                count = word_count.get(vocabword, 0)
                #概率公式：单词在该类别中得数量+lambda_a/此类别的总数量+lambda_a*词表大小
                probablity = (count + lambda_a) / (class_count[class_] + lambda_a * len(vocab_set))
                wordVsProb[vocabword] = probablity
        each_class_word_probablities[class_] = wordVsProb

import math
from sklearn.metrics import classification_report,confusion_matrix
def classify(each_class_word_probablities, words):
    class_results={}
    for class_,_ in each_class_word_probablities.items():
        probablity = 0
        for word in words:
            word_probablities = each_class_word_probablities.get(class_).get(word, None)
            if word_probablities != None:
                word_probablities = math.log(word_probablities)
                probablity += word_probablities
        class_results[class_] = probablity * 1/4
    sorted_class_list=sorted(class_results.items(),reverse=True,key = lambda kv:(kv[1], kv[0]))
    return sorted_class_list[0][0]


gold_label=[]
pre_label=[]
for each in test_data:
    result = classify(each_class_word_probablities, each[0])

    gold_label.append(each[1])
    pre_label.append(int(result))

print(classification_report(gold_label, pre_label))

print(confusion_matrix(gold_label, pre_label))


