from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import warnings
warnings.filterwarnings("ignore")
from transformers import BertTokenizer,RobertaTokenizer
import os
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from model import NLIModel
net = NLIModel()
net.to(device)
weights_path = "./bestModelbase.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
net.load_state_dict(torch.load(weights_path, map_location=device))
id2class={0:"business",1:"technology",2:"entertainment",3:"sports",4:"science",5:"health",}


class mainWindow(QWidget):
    def __init__(self):
        super(mainWindow, self).__init__()
        self.setWindowTitle("News Title Classification")
        self.resize(1100, 500)
        self.setStyleSheet("background-color:white")
        self.sourcefile_path = ""
        self.cwd = os.getcwd()

        self.label = QTextEdit(self)
        self.label.setFixedSize(900, 200)
        self.label.setPlaceholderText("Please enter the news title you want to predict")
        self.label.move(100, 60)

        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )
        tmp1 = QLabel(self)
        tmp1.setText("News Title Classification")
        tmp1.move(450,10)


        btn = QPushButton(self)
        btn.setText("Predict!")
        btn.move(480, 310)
        btn.clicked.connect(self.text2classify)


        self.label_10 = QLabel(self)
        self.label_10.setGeometry(QtCore.QRect(480, 370, 300, 60))
        self.label_10.setText("News Categories")

    def text2classify(self):
        text = self.label.toPlainText()

        text_encoded_text = tokenizer.encode_plus(
            text, add_special_tokens=True, truncation=True,
            max_length=64, padding='max_length',
            return_attention_mask=True,
            return_tensors='pt')

        test_text = text_encoded_text['input_ids'][0]
        test_text_mask = text_encoded_text['attention_mask'][0]
        outputs = net(test_text.unsqueeze(0).to(device), test_text_mask.unsqueeze(0).to(device))
        predict_y = torch.max(outputs.logits, dim=1)[1]
        result = predict_y.cpu().numpy().tolist()[0]
        self.label_10.setText("Result: "+str(id2class[int(result)]))



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainForm = mainWindow()
    mainForm.show()
    sys.exit(app.exec_())