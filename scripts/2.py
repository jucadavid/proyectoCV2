import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.cvtColor(cv2.imread("../train/1b4ccdf7d5ff45dc6c3885243bde5af2_jpg.rf.92bd5a7bb83cb5d8639db85d02fe7511.jpg"), cv2.COLOR_BGR2RGB)

template = cv2.cvtColor(cv2.imread("../templates/whiteKing.jpg"), cv2.COLOR_BGR2RGB)

print("Template Shape: ", template.shape)

mapa = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)

h, w = template.shape[:-1]


maxValue = max(mapa.flatten())
loc = np.where( mapa >= maxValue-5)

for i in range(0, len(loc), 2):
    print("Point:", loc[i][0], loc[i+1][0])
    cv2.rectangle(img, (loc[i+1][0], loc[i][0]), (loc[i+1][0]+w, loc[i][0]+h), (0, 0, 255), 2)

fig, (p1, p2, p3) = plt.subplots(1, 3)
p1.imshow(img)
p2.imshow(template)
p3.imshow(mapa, cmap="gray")

plt.show()
