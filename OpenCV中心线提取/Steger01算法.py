import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
def Eig2image(Dxx,Dxy,Dyy):
    tmp = np.sqrt(pow((Dxx - Dyy), 2) + 4 * Dxy * Dxy)
    v2x = 2 * Dxy
    v2y = Dyy - Dxx + tmp
    mag = np.sqrt(v2x*v2x+v2y*v2y)
    i = (mag!= 0)
    # 归一化
    v2x[i] = v2x[i]/mag[i]
    v2y[i] = v2y[i]/mag[i]
    # 正交
    v1x = -v2y
    v1y = v2x
    mu1 = 0.5*(Dxx+Dyy+tmp)
    mu2 = 0.5*(Dxx+Dyy-tmp)
    check = np.abs(mu1)>np.abs(mu2)
    Lambda1 = mu1
    Lambda1[check] = mu2[check]
    Lambda2 = mu2
    Lambda2[check] = mu1[check]
    lx = v1x
    lx[check] = v2x[check]
    ly =v1y
    ly[check] = v2y[check]
    return Lambda1,Lambda2,lx,ly

def Hessian(img,sigma= 8,size= 3):
    x = np.arange(-3*sigma,3*sigma-1)
    y = np.arange(-3*sigma,3*sigma-1)
    [X, Y] = np.meshgrid(x, y)
    DGaussx = 1 / (2 * np.pi *pow(sigma, 4)) * (-X) * np.exp(-(X * X + Y * Y) / (2 * sigma * sigma));
    DGaussy = 1 / (2 * np.pi *pow(sigma, 4)) * (-Y) * np.exp(-(X * X + Y * Y) / (2 * sigma * sigma));
    DGaussxx = 1 / (2 * np.pi *pow(sigma, 4)) * (X * X /pow(sigma, 2) - 1) * np.exp(
        -(X * X + Y * Y) / (2 * sigma * sigma));
    DGaussxy = 1 / (2 * np.pi *pow(sigma, 6)) * (X * Y) * np.exp(-(X * X + Y * Y) / (2 * sigma * sigma));
    DGaussyy = 1 / (2 * np.pi *pow(sigma, 4)) * (Y * Y /pow(sigma, 2) - 1) * np.exp(
        -(X * X + Y * Y) / (2 * sigma * sigma));
    print("DGaussx",DGaussx.shape)
    Dx = cv2.filter2D(img, -1, DGaussx)
    Dy = cv2.filter2D(img, -1, DGaussy)
    Dxx = cv2.filter2D(img, -1, DGaussxx)
    Dxy = cv2.filter2D(img, -1, DGaussxy)
    Dyy = cv2.filter2D(img, -1, DGaussyy)
    print("Dx.shape",Dx.shape)
    return Dx,Dy,Dxx,Dxy,Dyy
def Centerline(img,sigma,k):
    Dx,Dy,Dxx,Dxy,Dyy = Hessian(img,sigma)
    eigenvalue1, eigenvalue2, eigenvectorx, eigenvectory = Eig2image(Dxx,Dxy,Dyy)
    t =  -(Dx*eigenvectorx + Dy*eigenvectory)/(Dxx* eigenvectorx*eigenvectorx + 2*Dxy*eigenvectorx*eigenvectory + Dyy*eigenvectory*eigenvectory );
    px = t*eigenvectorx
    py = t*eigenvectory

    c = np.abs(px)<=k
    g = np.abs(py)<=k
    print("px",px)
    print("py",py)
    flags = np.bitwise_and(c,g)
    return flags
def nothing(pos):
    newimg = img.copy()
    thval = cv2.getTrackbarPos("thresh","origin")
    sigma = cv2.getTrackbarPos("sigma","origin")
    k = cv2.getTrackbarPos("k","origin")
    k = k/10.0
    gray = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
    np.savetxt("test.txt",gray,fmt= "%.3d")
    ret,thresh = cv2.threshold(gray,thval,255,cv2.THRESH_BINARY)
    gray = gray.astype(float)
    #gray = gray / 255
    start = cv2.getTickCount() 
    flags = Centerline(gray, sigma, k)
    print((cv2.getTickCount()-start)/cv2.getTickFrequency())
    print("dj", flags)
    flags = np.bitwise_and(flags,thresh>0)
    newimg[flags, :] = [0, 255, 0]
    cv2.imshow("res", newimg)
cv2.namedWindow("res",0)
img = cv2.imread("image/laser-v.jpg")
#img= img[:,1500:2500,:]
#cv2.namedWindow("origin",0)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img,(5,5),1)
cv2.imshow("origin",img)
#gray = gray.astype(float)
#gray = gray/255
cv2.createTrackbar("sigma","origin",0,20,nothing)
cv2.createTrackbar("thresh","origin",0,255,nothing)
cv2.createTrackbar("k","origin",1,10,nothing)

cv2.waitKey(0)


