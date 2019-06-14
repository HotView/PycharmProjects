import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

nrow = 58
ncol = 58
img = cv2.imread('meizu01.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow("win",0)
# 光度补偿
plt.imshow(img_gray, 'gray')
img_gray = np.tanh((img_gray.astype(np.float32) -
                    img_gray.mean()) / img_gray.std())
img_gray = cv2.dilate(img_gray, None) + cv2.erode(img_gray, None)
plt.figure(2)
plt.imshow(img_gray, 'gray')
Ix = cv2.Scharr(img_gray, cv2.CV_32F, 1, 0)
Iy = cv2.Scharr(img_gray, cv2.CV_32F, 0, 1)
Ixx = cv2.Scharr(Ix, cv2.CV_32F, 1, 0)
Ixy = cv2.Scharr(Ix, cv2.CV_32F, 0, 1)
Iyy = cv2.Scharr(Iy, cv2.CV_32F, 0, 1)
Iyx = cv2.Scharr(Iy, cv2.CV_32F, 1, 0)
fig = plt.figure(3)
ax1 = fig.add_subplot(221)
ax1.imshow(Ix, 'gray')
ax2 = fig.add_subplot(222)
ax2.imshow(Iy, 'gray')
ax3 = fig.add_subplot(223)
ax3.imshow(Ixy, 'gray')

detector = (cv2.GaussianBlur(Ixy, (5, 5), 0) *
            cv2.GaussianBlur(Iyx, (5, 5), 0) -
            cv2.GaussianBlur(Ixx, (5, 5), 0) *
            cv2.GaussianBlur(Iyy, (5, 5), 0))
print("detector MAX:", np.amax(detector))
print("detector MAX:", np.amin(detector))

plt.figure(4)
plt.imshow(detector)

kernel = np.array([[-3, -2, -1, 0, 1, 2, 3],
                   [-2, -2, -1, 0, 1, 2, 2],
                   [-1, -1, -1, 0, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 1, 0, -1, -1, -1],
                   [2, 2, 1, 0, -1, -2, -2],
                   [3, 2, 1, 0, -1, -2, -3]],
                  dtype=np.float32)

detector *= cv2.filter2D(img_gray, cv2.CV_32F, kernel)
detector /= np.sqrt(np.mean(np.square(detector)))
pdetector = np.fmax(0, detector)
ndetector = np.fmax(0, -detector)
pdetector, ndetector = (
    np.fmax(pdetector - cv2.GaussianBlur(ndetector * 10, (7, 7), 0), 0),
    np.fmax(ndetector - cv2.GaussianBlur(pdetector * 10, (7, 7), 0), 0))

# find x-corner candidates
print(np.amax(pdetector), np.amin(pdetector),
      np.amax(ndetector), np.amin(ndetector))
plt.figure()
plt.imshow(pdetector)
plt.figure()
plt.imshow(ndetector)
points1 = cv2.goodFeaturesToTrack(
    pdetector, maxCorners=(nrow + 1) * (ncol + 1),
    qualityLevel=0.1, minDistance=5, blockSize=3)[:, 0, :]
points2 = cv2.goodFeaturesToTrack(
    ndetector, maxCorners=(nrow + 1) * (ncol + 1),
    qualityLevel=0.1, minDistance=5, blockSize=3)[:, 0, :]
print("points1 shape:")
print(points2.shape)
print(points2[0])
# pick a start point
center = points1[np.argmin(np.sum(np.square(
    points1 - np.median(min(points1, points2, key=len), axis=0)), axis=1))]
print("-------------------------------")
print(np.square(points1 - np.median(min(points1, points2, key=len), axis=0)).shape)
print("center information")
print(np.argmin(np.sum(np.square(
    points1 - np.median(min(points1, points2, key=len), axis=0)), axis=1)))
print("center shape:", center.shape)
print("center:", center)
j, i = np.round(center).astype(int)
print(i, j)
V = np.linalg.eig(np.array([[Ixx[i, j], Ixy[i, j]],
                            [Iyx[i, j], Iyy[i, j]]]))[1]
print("V.shape:", V.shape)
print("V", V)
V = np.dot([[1, 1], [1, -1]], V.T)
print("V.shape:", V.shape)
print("V", V)
V /= np.hypot(V[:, 0], V[:, 1])
print("V.shape:", V.shape)
print("V", V)
# find nearest x-corner along row and column
print("-----------------------------------------")
print("abs shape:", np.abs(np.dot(points2 - center, V)).shape)
D1, D2 = np.abs(np.dot(points2 - center, V)).T
# print("D1和D2",D1,D2)
print("D1和D2de shape", D1.shape, D2.shape)
D = D1 + D2
print("D.shape", D.shape)
point1 = points2[np.argmin(np.where((D1 < D2), D, np.inf))]
point2 = points2[np.argmin(np.where((D2 < D1), D, np.inf))]
print("--------------------------------------")
print("point1", point1)
print("point2", point2)
# calculate row and column offset near center point
diamond = np.array([point1, center * 2 - point1,
                    point2, center * 2 - point2])
print(diamond.argmax(axis=0))
offset = (diamond[diamond.argmax(axis=0)] -
          diamond[diamond.argmin(axis=0)]) / 2
print("diamond", type(diamond))
print(diamond.shape)
print(diamond)
print(offset.shape)
print("offset", type(offset))
print(offset)
# generate window for point refining (empirical size)
d_scan = np.reshape(np.meshgrid(np.arange(-3, 4),
                                np.arange(-3, 4)), (2, -1))
print("d_scan", d_scan)
dX_scan, dY_scan = d_scan[:, np.sum(np.abs(d_scan), axis=0) <= 4]
print("Dx_scan", dX_scan, "DY_scan", dY_scan)
# improve gradient for point refining (size is insignificant)
pdetector += cv2.GaussianBlur(pdetector, (25, 25), 0)
ndetector += cv2.GaussianBlur(ndetector, (25, 25), 0)

coeffs_new = np.reshape(np.meshgrid([-1, 0, 1], [-1, 0, 1]), (2, -1)).T
points_new = np.dot(coeffs_new, offset) + center
print("----------------")
print("coeffs_new dot offset")
print(np.dot(coeffs_new, offset))


nrow = 58
ncol = 58

cell_width = 588.1 / 58000  # m, horizontal length of grid
cell_height = 580.95 / 58000  # m, vertical length of grid


print('Loading files...')
print(np.linspace(-ncol, ncol, ncol + 1))
print(cell_width)
print(np.linspace(-ncol, ncol, ncol + 1) * cell_width / 2)
chessboard_corners_3d = np.hstack([np.reshape(np.meshgrid(
    np.linspace(-ncol, ncol, ncol + 1) * cell_width / 2,
    np.linspace(-nrow, nrow, nrow + 1) * cell_height / 2),
    (2, -1)).T, np.zeros(((ncol + 1) * (nrow + 1), 1))]).astype(np.float32)
print("chessboard_corners_3d.shape", chessboard_corners_3d.shape)
print(chessboard_corners_3d, "###############")

# object_points = [chessboard_corners_3d] * len(image_points)
object_points = [chessboard_corners_3d] * 10
print(type(object_points))
print("object_points.shape", len(object_points))
print(object_points)

print('Calibrating 相机视频demo...')
actions = [lambda w, n, e, s: ((w - 1, n, e, s),
                               np.transpose([np.full(s - n + 1, w - 1),
                                             np.arange(n, s + 1)])),
           lambda w, n, e, s: ((w, n - 1, e, s),
                               np.transpose([np.arange(w, e + 1),
                                             np.full(e - w + 1, n - 1)])),
           lambda w, n, e, s: ((w, n, e + 1, s),
                               np.transpose([np.full(s - n + 1, e + 1),
                                             np.arange(n, s + 1)])),
           lambda w, n, e, s: ((w, n, e, s + 1),
                               np.transpose([np.arange(w, e + 1),
                                             np.full(e - w + 1, s + 1)]))]
action = None
bound_new = -1, -1, 1, 1
coeffs_new = np.reshape(np.meshgrid([-1, 0, 1], [-1, 0, 1]), (2, -1)).T
i = 0
while(True):
    i = i + 1
    umin, vmin, umax, vmax = bound_new
    action = actions[0 if action not in actions else
                     actions.index(action) - 1]
    bound_new, coeffs_new = action(umin, vmin, umax, vmax)
    print(coeffs_new)
    if i > 6:
        break
