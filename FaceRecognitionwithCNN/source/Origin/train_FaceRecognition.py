import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image

TRAIN_DATA = 'FaceRecognitionwithCNN/dataset/train_data'
TEST_DATA = 'FaceRecognitionwithCNN/dataset/test_data'

#Ví dụ: classes = ['Justin Bieber', 'Ronaldo', 'G-Dragon', 'Reus', 'Son Tung', 'David Beckham', 'Messi']
classes = []
xtrain = []
ytrain = []
xtest = []
ytest = []
dictionary = {}
np.random.seed(5)


#Get Dictionary from Data
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


#Get Data
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
classes = list(dictionary.keys())


xtrain, list_train = Data(TRAIN_DATA, dictionary).operate()
xtest, list_test = Data(TEST_DATA, dictionary).operate()


xtrain, xtest = np.array(xtrain)/100, np.array(xtest)/100
ytrain, ytest = np.array([np.expand_dims(item, axis=0) for item in list_train]), np.array([np.expand_dims(item, axis=0) for item in list_test])


#Chuyển đổi label về dạng One-hot Encoding để đưa qua Softmax
ytrain_ohe, ytest_ohe = to_categorical(ytrain), to_categorical(ytest)


#Có thể thêm vài vòng lặp nữa để dữ liệu đoán chính xác hơn Loop(Conv2D,MaxPooling,Dropout)
#Model sử dụng Kernel 3x3
model_training = models.Sequential([
    layers.Conv2D(32,(3,3), input_shape=(128,128,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.15),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(len(classes), activation='softmax')
])




model_training.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
history = model_training.fit(xtrain,ytrain_ohe,epochs=10,validation_data=(xtest,ytest_ohe))
model_training.save('FaceRecognitionwithCNN/model-faceRecognition.v11')



#Vẽ biểu đồ của accuracy và val_accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('FaceRecognitionwithCNN/source/Origin/Picture/Statistical_chart_train_model_v11.jpg')
test_loss, test_acc = model_training.evaluate(xtest, ytest_ohe, verbose=2)
plt.show()

