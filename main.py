import cv2
import numpy as np
import matplotlib.pyplot as plt

trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
# random màu (1-Đỏ, 2-xanh)
ketqua = np.random.randint(0, 2, (25, 1)).astype(np.float32)
red = trainData[ketqua.ravel()==1]
blue = trainData[ketqua.ravel()==0]
newMember = np.random.randint(0, 100, (1, 2)).astype(np.float32)
# vẽ toạ độ
# toạ độ x, toạ độ y, độ lớn, màu sắc, hiình dạng
plt.scatter(red[:,0], red[:,1], 100, 'r', 's')
plt.scatter(blue[:,0], blue[:,1], 100, 'b', '^')
plt.scatter(newMember[:,0], newMember[:,1], 100, 'g', 'o')

knn = cv2.ml.KNearest_create()
knn.train(trainData, 0, ketqua)
temp, ketqua, hangxom, khoangcach = knn.findNearest(newMember,3)

print("Result: {}\n".format(ketqua))
print("Hang Xom: {}\n".format(hangxom))
print("Khoang Cach: {}\n".format(khoangcach))

plt.show()