import numpy as np
import os
import cv2
from tensorflow.keras import models
from PIL import Image
TRAIN_DATA = 'FaceRecognitionwithCNN/dataset/train_data'
face_detector = cv2.CascadeClassifier('FaceRecognitionwithCNN/haarcascades/haarcascade_frontalface_alt.xml')
model_training = models.load_model('FaceRecognitionwithCNN/model-faceRecognition.v1')
dictionary = {}

#Đường dẫn: Nếu tồn tại file.mp4, file.jpg, file.png sẽ được nhận dạng tương ứng, Nếu không sẽ lấy hình ảnh từ Camera
path = 'FaceRecognitionwithCNN/test/ronaldo.mp4'



class Data:
    def __init__(self, path):
        self.path = path
        self.dictionary = {}
    
    def operate(self):
        number = 0
        for whatever in os.listdir(self.path):
            if(whatever != '.DS_Store'):
                self.dictionary[whatever] = number
                number += 1
        return self.dictionary


dictionary = Data(TRAIN_DATA).operate()
classes = list(dictionary.keys())



class Recognition:
    def __init__(self, path):
        self.path = path

    def Video_Recognition(self, value):
        if value == 1:
            cam = cv2.VideoCapture(self.path)
        else:
            cam = cv2.VideoCapture(0)
        while True:
            Ok, frame = cam.read()
            faces = face_detector.detectMultiScale(frame, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                roi = cv2.resize(frame[y:y+h, x:x+w],(128,128))
                result = np.argmax(model_training.predict(roi.reshape((-1,128,128,3))))
                # print(result)
                cv2.putText(frame, classes[result],(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()

    def Image_Recognition(self):
        image = cv2.imread(self.path)
        if image is not None:
            faces = face_detector.detectMultiScale(image, 1.3, 5)
            count = 1
            if(count <= len(faces)):
                for (x,y,w,h) in faces:
                    roi = cv2.resize(image[y:y+h, x:x+w],(128,128))
                    # print(model_training.predict(roi.reshape((-1,128,128,3))))
                    result = np.argmax(model_training.predict(roi.reshape((-1,128,128,3))))
                    cv2.putText(image, classes[result],(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
                    cv2.rectangle(image,(x,y),(x+w,y+h), (0,255,0),1 )
                    count += 1
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('Failed to read the image.')

    def operate(self):
        if os.path.exists(self.path):
            if(self.path.endswith(('.jpg','.png'))):
                print('Input from Image')
                self.Image_Recognition()
            elif(self.path.endswith('.mp4')):
                print('Input from Video')
                self.Video_Recognition(1)
        else:
            print('Direc: ' + self.path + ' No Exists!' '\nInput from Camera')
            self.Video_Recognition(0)



Recognition(path).operate()

