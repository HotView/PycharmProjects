from scipy.optimize import leastsq
def residuals(p,Y,X):
    k,b = p
    return Y-(k*X+b)
def getFitline(points):
    """
    :param points:2-D ndarray
    :return:k,b
    """
    X = points[:,0]
    Y = points[:,1]
    p = [-2.,0]
    r = leastsq(residuals,p,args=(Y,X))
    return r[0]
