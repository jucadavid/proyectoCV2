import matplotlib.pyplot as plt
import numpy as np
import cv2

img_rgb = cv2.cvtColor(cv2.imread("../train/1b4ccdf7d5ff45dc6c3885243bde5af2_jpg.rf.92bd5a7bb83cb5d8639db85d02fe7511.jpg"), cv2.COLOR_BGR2RGB)

img_R = img_rgb[:,:,0]
img_G = img_rgb[:,:,1]
img_B = img_rgb[:,:,2]
#Se obtienen los canales YIQ mediante una transformación lineal
img_Y = 0.299*img_R + 0.587*img_G + 0.114*img_B
img_I = 0.596*img_R - 0.274*img_G - 0.322*img_B
img_Q = 0.211*img_R - 0.523*img_G + 0.312*img_B

img_yiq = np.zeros(img_rgb.shape)

img_yiq[::,::,0] = img_Y
img_yiq[::,::,1] = img_I
img_yiq[::,::,2] = img_Q

fig, (p1, p2, p3) = plt.subplots(1, 3)
p1.imshow(img_Y)
p2.imshow(img_I)
p3.imshow(img_Q)
plt.show()
