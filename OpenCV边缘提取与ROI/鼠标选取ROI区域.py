import cv2
import numpy as np

lsPointsChose= []
tpPointsChoose = [] # 存入选择的点
pointsCount = 0 #对鼠标按下的点计数
pointsMax = 2
def ROI_byMouse():
    global src, ROI, ROI_flag, mask2
    mask = np.zeros(img.shape, np.uint8)
    pts = np.array([lsPointsChose], np.int32)  # pts是多边形的顶点列表（顶点集）
    pts = pts.reshape((-1, 1, 2))
    # 这里 reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
    # OpenCV中需要先将多边形的顶点坐标变成顶点数×1×2维的矩阵，再来绘制

    # --------------画多边形---------------------
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    ##-------------填充多边形---------------------
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    cv2.imshow('mask', mask2)
    cv2.imwrite('mask.bmp', mask2)
    ROI = cv2.bitwise_and(mask2, img)
    cv2.imwrite('ROI.bmp', ROI)
    cv2.imshow('ROI', ROI)
def on_mouse(event,x,y,flags,param):
    global img,point1,point2,count,pointsMax
    global lsPointsChose,tpPointsChoose
    global pointsCount
    global img2,ROI_bymouse_flag
    img2 = img.copy
    if event == cv2.EVENT_LBUTTONDOWN:
        pointsCount+=1
        print("poinsCount:",pointsCount)
        point1 = (x,y)
        print("position:",x,y)
        cv2.circle(img2,point1,10,(0,255,0),2)
        # 将选取的点保存到list列表里
        lsPointsChose.append([x,y])# 用于转化为ndarry，提取多边形ROI
        tpPointsChoose.append((x,y))# 用于画点
        # 将鼠标的点用直线连接起来
        print(len(tpPointsChoose))
        for i in range(len(tpPointsChoose)-1):
            print("i",i)
            cv2.line(img2,tpPointsChoose[i],tpPointsChoose[i+1],(0,0,255),2)
        if (pointsCount == pointsMax):
            # -----------绘制感兴趣区域-----------
            ROI_byMouse()
            ROI_bymouse_flag = 1
            lsPointsChoose = []

        cv2.imshow('src', img2)
img = cv2.imread('chessimg/laser01.jpg')
# ---------------------------------------------------------
# --图像预处理，设置其大小
# height, width = img.shape[:2]
# size = (int(width * 0.3), int(height * 0.3))
# img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
# ------------------------------------------------------------
ROI = img.copy()
cv2.namedWindow('src')
cv2.setMouseCallback('src', on_mouse)
cv2.imshow('src', img)
cv2.waitKey(0)
cv2.threshold(img, )
img.coyto()