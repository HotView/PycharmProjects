import numpy as np
import cv2
img = cv2.imread("test_rec.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("test", 0)
cv2.namedWindow("detector", 0)
Ix = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
Iy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
Ixx = cv2.Scharr(Ix, cv2.CV_32F, 1, 0)
Ixy = cv2.Scharr(Ix, cv2.CV_32F, 0, 1)
Iyy = cv2.Scharr(Iy, cv2.CV_32F, 0, 1)
Iyx = cv2.Scharr(Iy, cv2.CV_32F, 1, 0)
detector = (cv2.GaussianBlur(Ixy, (5, 5), 0) * cv2.GaussianBlur(Iyx, (5, 5), 0) -
            cv2.GaussianBlur(Ixx, (5, 5), 0) * cv2.GaussianBlur(Iyy, (5, 5), 0))
points = cv2.goodFeaturesToTrack(
    detector, maxCorners=36, qualityLevel=0.1, minDistance=5, blockSize=3)
points_int = points.astype(np.int).reshape((-1, 2))
point_media = np.median(points_int, axis=0).astype(np.int)
points_sum = np.sum(np.square(points_int - point_media), axis=1)
index_min = np.argmin(points_sum)
print(index_min)
center = points_int[index_min]
i, j = center
img[i, j, :] = [255, 0, 0]
V = np.linalg.eig(np.array([[Ixx[i, j], Ixy[i, j]], [Iyx[i, j], Iyy[i, j]]]))[1]
V_change = np.dot([[1, 1], [1, -1]], V.T)
V_change = (V_change * 10).astype(np.int)
V = V * 10
V = V.astype(np.int)
cv2.arrowedLine(img, (center[0], center[1]), (center[0] +
                                              V[0, 0], center[1] + V[0, 1]), (0, 0, 255), 1)
cv2.arrowedLine(img, (center[0], center[1]), (center[0] +
                                              V[1, 0], center[1] + V[1, 1]), (0, 255, 0), 1)
cv2.arrowedLine(img, (center[0], center[1]), (center[0] + V_change[0, 0], center[1] + V_change[0, 1]), (0, 0, 255), 2)
cv2.arrowedLine(img, (center[0], center[1]), (center[0] + V_change[1, 0], center[1] + V_change[1, 1]), (0, 255, 0), 2)

print(type(points_int - center))
print((points_int - center).shape)
Dx, Dy = (points_int - center).T
D = np.abs(Dx) + np.abs(Dy)
np_where = np.where(np.all([Dx > 0, Dx > Dy], axis=0), D, np.inf)
print('----------------')
point1 = points_int[np.argmin(np_where)]
img[point1[0], point1[1], :] = [255, 0, 0]
print(point1[0], point1[1])
center = point1
i, j = center
print(i, j, "ij")
V = np.linalg.eig(np.array([[Ixx[i, j], Ixy[i, j]], [Iyx[i, j], Iyy[i, j]]]))[1]
V_change = np.dot([[1, 1], [1, -1]], V.T)
V_change = (V_change * 10).astype(np.int)
V = V * 10
V = V.astype(np.int)
print(i, j)
cv2.arrowedLine(img, (i, j), (center[0] + V[0, 0], center[1] + V[0, 1]), (0, 0, 255), 1)
cv2.arrowedLine(img, (j, i), (center[0] + V[1, 0], center[1] + V[1, 1]), (0, 255, 0), 1)
# point1_index,point2_index = np.argmin(np.where((Dx>0 and Dx<Dy),D,np.inf))
# print(point1_index,point2_index)

img[160, :, :] = [0, 0, 255]
cv2.imshow("test", img)
cv2.imshow("detector", detector)
cv2.waitKey()
