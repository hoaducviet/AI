import numpy as np
import os
import cv2

from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image


TRAIN_DATA = 'FaceRecognitionwithCNN/dataset/train_data'
TEST_DATA = 'FaceRecognitionwithCNN/dataset/test_data'
#classes mẫu như bên dưới, nếu dùng bộ dữ liệu khác thì classes sẽ khác, tự động thay đổi theo dữ liệu đầu vào
# classes = ['Justin Bieber', 'Ronaldo', 'G-Dragon', 'Reus', 'Son Tung', 'David Beckham', 'Messi']
classes = []
xtrain = []
ytrain = []
xtest = []
ytest = []
dictionary = {}



class Dictionary:
    def __init__(self, path):
        self.path = path
        self.dictionary = {}
    def operate(self):
        number = 0
        for name in os.listdir(self.path):
            if not name.startswith('.'):
                self.dictionary[name] = number
                number += 1
        return self.dictionary



class Data:
    def __init__(self, path, dictionary):
        self.path = path
        self.dictionary = dictionary
        self.input = []
        self.label = []
        self.listData = []

    def getData(self):
        for whatever in os.listdir(self.path):
            if(whatever != '.DS_Store'):
                whatever_path = os.path.join(self.path, whatever)
                list_data_filename_path = []
                for filename in os.listdir(whatever_path):
                    if(filename != '.DS_Store'):
                        filename_path = os.path.join(whatever_path, filename)
                        label = filename_path.split('/')[3]
                        img = np.array(Image.open(filename_path))
                        list_data_filename_path.append((img,self.dictionary[label]))
                self.listData.extend(list_data_filename_path)
        return self.listData
    
    def operate(self):
        self.listData = self.getData()
        np.random.shuffle(self.listData)
        for Data in self.listData:
            self.input.append(np.array(Data[0])) 
            self.label.append(Data[1])
        
        return self.input, self.label



dictionary = Dictionary(TRAIN_DATA).operate()
xtrain, list_train = Data(TRAIN_DATA, dictionary).operate()
xtest, list_test = Data(TEST_DATA, dictionary).operate()
xtrain, xtest = np.array(xtrain)/100, np.array(xtest)/100
ytrain, ytest = np.array([np.expand_dims(item, axis=0) for item in list_train]), np.array([np.expand_dims(item, axis=0) for item in list_test])
classes = list(dictionary.keys())



# Test số hình ảnh đoán đúng từ dataset/test_data
model_training = models.load_model('FaceRecognitionwithCNN/model-faceRecognition.v1')
acc = 0
fig = plt.figure(figsize=(15,15))
for i in range(50):
    plt.subplot(5,10,i+1)
    #Vì dataset/train chưa đủ lớn, đa dạng nên để xtest thì tỷ lệ đúng ít, thử thay xtest, ytest thành xtrain, ytrain (tỉ lệ đúng khá cao)
    plt.imshow(xtrain[i])
    result = np.argmax(model_training.predict(xtrain[i].reshape((-1,128,128,3))))
    if result == ytrain[i][0]:
        acc +=1
    plt.title(classes[result])
    plt.grid(color='lightgray', linestyle='--')
    plt.axis("off")
    plt.savefig('FaceRecognitionwithCNN/source/Origin/Picture/Test_results_model_v11.jpg')

print(acc)
plt.show()