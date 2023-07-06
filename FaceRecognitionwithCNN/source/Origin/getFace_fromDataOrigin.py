import cv2
import os


face_detector = cv2.CascadeClassifier('FaceRecognitionwithCNN/haarcascades/haarcascade_frontalface_alt.xml')



def getFace_fromImage(dataImage_path,faceRec_path,count):
    print(dataImage_path + " to " + faceRec_path)
    img = cv2.imread(dataImage_path)
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        img_face = cv2.resize(img[y+1:y+h-1,x+1:x+w-1],(128,128))
        # print(img_face.shape)
        count += 1
        cv2.imwrite(faceRec_path+'/pic_{}.jpg'.format(count), img_face)
    #     cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)
    #     cv2.imshow('Image',img)
    # cv2.destroyAllWindows()
    return count
    


def getFace_fromVideo(dataVideo_path,faceRec_path,count):
    print(dataVideo_path + " to " + faceRec_path)
    cap = cv2.VideoCapture(dataVideo_path)

    if not cap.isOpened():
        print(f"Can't Open {dataVideo_path}")
        return
    while True:
        Ok, frame = cap.read()
        if not Ok:
            break
        faces = face_detector.detectMultiScale(frame, 1.3, 5)
        for (x,y,w,h) in faces:
            roi = cv2.resize(frame[y+1:y+h-1, x+1:x+w-1],(128,128))
            print(roi.shape)
            count += 1
            cv2.imwrite(faceRec_path + '/pic_{}.jpg'.format(count), roi)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return count




dataTrain_path = 'FaceRecognitionwithCNN/dataOrigin/train_data'
dataTest_path = 'FaceRecognitionwithCNN/dataOrigin/test_data'



def getData(path):
    numberClass = 0
    for whatelse in os.listdir(path):
        if(whatelse != '.DS_Store'):
            whatelse_path = os.path.join(path,whatelse)
            count = 0
            faceRec_path = whatelse_path.replace('dataOrigin','dataset')
            if not os.path.isdir(whatelse_path.replace('dataOrigin','dataset')):
                os.mkdir(whatelse_path.replace('dataOrigin','dataset'))
            for sub_whatelse in os.listdir(whatelse_path):
                if(sub_whatelse != '.DS_Store'):
                    data_path = os.path.join(whatelse_path,sub_whatelse)
                    if data_path.endswith('.mp4'):
                        count = getFace_fromVideo(data_path,faceRec_path,count)
                    if data_path.endswith(('.jpg', '.png', '.jpeg')):
                        count = getFace_fromImage(data_path,faceRec_path,count)
            print(count)
getData(dataTrain_path)
getData(dataTest_path)