import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('FaceRecognitionwithCNN/test/image.jpg')
img = cv2.resize(img, (200,200))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255

# input.shape tra ve gia tri kich thuoc anh

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
    
img_gray_conv2d = Conv2d(img_gray,16,3, padding=0, stride=1).operate()


fig = plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(img_gray_conv2d[:,:,i],cmap="gray")
    plt.axis("off")
plt.savefig('FaceRecognitionwithCNN/source/Layer/Picture/Img_gray_conv2d.jpg')
plt.show()
 


