import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('FaceRecognitionwithCNN/test/image.jpg')
img = cv2.resize(img, (200,200))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255


np.random.seed(5)
class Conv2d:
    def __init__(self, input,numOfKernel=8, kernelSize=3, padding=0, stride=1):
        self.input = np.pad(input, ((padding,padding),(padding,padding)), "constant")
        self.kernel = np.random.randn(numOfKernel, kernelSize, kernelSize)
        self.stride = stride
        self.result = np.zeros((int((self.input.shape[0] - self.kernel.shape[1])/self.stride) + 1, int((self.input.shape[1] - self.kernel.shape[2])/self.stride) + 1, self.kernel.shape[0]))

    def getROI(self): #Region of interesting (Vung du lieu quan tam)
        for row in range(int((self.input.shape[0] - self.kernel.shape[1])/self.stride) + 1):
            for col in range(int((self.input.shape[1] - self.kernel.shape[2])/self.stride) + 1):
                roi = self.input[row*self.stride: row*self.stride + self.kernel.shape[1], col*self.stride: col*self.stride + self.kernel.shape[2]]
                yield row, col, roi
    
    def operate(self):
        for layer in range(self.kernel.shape[0]):
            for row, col, roi in self.getROI():
                self.result[row,col,layer] = np.sum(roi * self.kernel[layer,:,:])  #luu y ve nhan 2 ma tran

        return self.result
    
#Giá trị các shape của kernel tạo bởi np lần lượt [layer,height,width]
#Giá trị các shape của img=[height,width,layer]


class Relu:
    def __init__(self, input):
        self.input = input
        self.result = np.zeros((self.input.shape[0], self.input.shape[1],self.input.shape[2]))

    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row,col,layer] = 0 if self.input[row,col,layer] < 0 else self.input[row,col,layer]
        return self.result



class leakyRelu:
    def __init__(self, input):
        self.input = input
        self.result = np.zeros((self.input.shape[0], self.input.shape[1],self.input.shape[2]))

    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row,col,layer] = 0.1*self.input[row,col,layer] if self.input[row,col,layer] < 0 else self.input[row,col,layer]
        return self.result



class MaxPooling:
    def __init__(self,input,poolingSize=2):
        self.input = input
        self.poolingSize = poolingSize
        self.result = np.zeros((int(self.input.shape[0]/self.poolingSize),int(self.input.shape[1]/self.poolingSize),self.input.shape[2]))
    
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(int(self.input.shape[0]/self.poolingSize)):
                for col in range(int(self.input.shape[1]/self.poolingSize)):
                    self.result[row,col,layer] = np.max(self.input[row*self.poolingSize: row*self.poolingSize + self.poolingSize, col*self.poolingSize: col*self.poolingSize + self.poolingSize, layer])
        return self.result



img_gray_conv2d = Conv2d(img_gray,16,3, padding=0, stride=1).operate()
img_gray_conv2d_relu = leakyRelu(img_gray_conv2d).operate()
img_gray_conv2d_relu_maxpooling = MaxPooling(img_gray_conv2d_relu).operate()


fig = plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(img_gray_conv2d_relu_maxpooling[:,:,i],cmap="gray")
    plt.axis("off")
plt.savefig('FaceRecognitionwithCNN/source/Layer/Picture/Img_gray_conv2d_leakyrelu_maxpooling.jpg')
plt.show()
 