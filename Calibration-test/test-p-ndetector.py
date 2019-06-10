import cv2
import numpy as np
import matplotlib.pyplot as plt
def find_large_chessboard(gray, nrow=58, ncol=58):

    # fix underexposure and overexposure
    gray = np.tanh((gray.astype(np.float32) - gray.mean()) / gray.std())
    gray = cv2.dilate(gray, None) + cv2.erode(gray, None)

    # calculate x-corner detector Ixy * Iyx - Ixx * Iyy
    #     reference:
    #         Gustavo Teodoro Laureano, etc. Topological Detection of
    #         Chessboard Pattern for 相机视频demo Calibration.
    #plt.imshow(gray,cmap="gray")
    np.savetxt(file, gray, fmt="%.3e")
    cv2.imshow("imggray",gray)
    cv2.normalize(gray,gray,0,1,cv2.NORM_MINMAX)
    np.savetxt("dataf.txt",gray)
    print(gray.dtype)
    cv2.imshow("imgflaot",gray)

    Ix = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    Iy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    Ixx = cv2.Scharr(Ix, cv2.CV_32F, 1, 0)
    Ixy = cv2.Scharr(Ix, cv2.CV_32F, 0, 1)
    Iyx = cv2.Scharr(Iy, cv2.CV_32F, 1, 0)
    Iyy = cv2.Scharr(Iy, cv2.CV_32F, 0, 1)
    detector = (cv2.GaussianBlur(Ixy, (5, 5), 0) *
                cv2.GaussianBlur(Iyx, (5, 5), 0) -
                cv2.GaussianBlur(Ixx, (5, 5), 0) *
                cv2.GaussianBlur(Iyy, (5, 5), 0))  # empirical size

    # split x-corners into two groups to reduce matching errors:
    #
    #     P :  ####        N :     ####
    #          ####                ####
    #              ####        ####
    #              ####        ####
    #
    kernel = np.array([[-3, -2, -1,  0,  1,  2,  3],
                       [-2, -2, -1,  0,  1,  2,  2],
                       [-1, -1, -1,  0,  1,  1,  1],
                       [0,  0,  0,  0,  0,  0,  0],
                       [1,  1,  1,  0, -1, -1, -1],
                       [2,  2,  1,  0, -1, -2, -2],
                       [3,  2,  1,  0, -1, -2, -3]],
                      dtype=np.float32)  # empirical size
    gray_kernel= cv2.filter2D(gray, cv2.CV_32F, kernel)
    np.savetxt("datagk.txt",gray_kernel,fmt="%3.e")
    detector *=gray_kernel
    detector /= np.sqrt(np.mean(np.square(detector)))
    pdetector = np.fmax(0, detector)
    ndetector = np.fmax(0, -detector)
    print(np.min(ndetector))
img = cv2.imread("test.png")
file = open("data.txt",mode='w')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
find_large_chessboard(gray,nrow = 6,ncol=4)
file.close()
plt.show()
cv2.waitKey(0)
