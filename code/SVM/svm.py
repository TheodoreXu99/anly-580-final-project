import pdb

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from random import shuffle
from sklearn.metrics import classification_report,confusion_matrix
text_features=torch.load("text_features.pt")
shuffle(text_features)
data=[each[1][0].numpy().tolist() for each in text_features]
target=[each[0] for each in text_features]
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.1)


svm = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0,\
          shrinking=True, probability=False, tol=0.001, cache_size=200,\
          class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
svm.fit(data_train, target_train)


target_pred = svm.predict(data_test)


true = np.sum(target_pred == target_test )
print(confusion_matrix(target_test,target_pred))
print(classification_report(target_test,target_pred))
