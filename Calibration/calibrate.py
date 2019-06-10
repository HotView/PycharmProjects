#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: eph

from __future__ import division

import os

import cv2
import numpy as np

from .config import board_rotation

__version__ = '2.2.180420'


# fix Chinese path

def cv2_imread(path):
     return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)


def cv2_imwrite(path, image):
     return cv2.imencode(os.path.splitext(path)[1], image)[1].tofile(path)


def find_large_chessboard(gray, nrow=58, ncol=58):

    # fix underexposure and overexposure
    gray = np.tanh((gray.astype(np.float32) - gray.mean()) / gray.std())
    gray = cv2.dilate(gray, None) + cv2.erode(gray, None)

    # calculate x-corner detector Ixy * Iyx - Ixx * Iyy
    #     reference:
    #         Gustavo Teodoro Laureano, etc. Topological Detection of
    #         Chessboard Pattern for 相机视频demo Calibration.
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
    detector *= cv2.filter2D(gray, cv2.CV_32F, kernel)
    detector /= np.sqrt(np.mean(np.square(detector)))
    pdetector = np.fmax(0, detector)
    ndetector = np.fmax(0, -detector)
    pdetector, ndetector = (  # empirical size
        np.fmax(pdetector - cv2.GaussianBlur(ndetector * 10, (7, 7), 0), 0),
        np.fmax(ndetector - cv2.GaussianBlur(pdetector * 10, (7, 7), 0), 0))

    # find x-corner candidates
    points1 = cv2.goodFeaturesToTrack(  # empirical configuration
        pdetector, maxCorners=(nrow + 1) * (ncol + 1),
        qualityLevel=0.1, minDistance=5, blockSize=3)[:, 0, :]
    points2 = cv2.goodFeaturesToTrack(  # empirical configuration
        ndetector, maxCorners=(nrow + 1) * (ncol + 1),
        qualityLevel=0.1, minDistance=5, blockSize=3)[:, 0, :]

    # pick a start point
    center = points1[np.argmin(np.sum(np.square(points1 - np.median(min(points1, points2, key=len), axis=0)), axis=1))]

    # find approximate row and column direction base on Hessian matrix
    #     as gradient change fastest along row and column
    j, i = np.round(center).astype(int)
    V = np.linalg.eig(np.array([[Ixx[i, j], Ixy[i, j]],
                                [Iyx[i, j], Iyy[i, j]]]))[1]
    V = np.dot([[1, 1], [1, -1]], V.T)
    V /= np.hypot(V[:, 0], V[:, 1])

    # find nearest x-corner along row and column
    D1, D2 = np.abs(np.dot(points2 - center, V)).T
    D = D1 + D2
    point1 = points2[np.argmin(np.where((D1 < D2), D, np.inf))]
    point2 = points2[np.argmin(np.where((D2 < D1), D, np.inf))]

    # calculate row and column offset near center point
    diamond = np.array([point1, center * 2 - point1,
                        point2, center * 2 - point2])
    offset = (diamond[diamond.argmax(axis=0)] -
              diamond[diamond.argmin(axis=0)]) / 2

    # generate window for point refining (empirical size)
    d_scan = np.reshape(np.meshgrid(np.arange(-3, 4),
                                    np.arange(-3, 4)), (2, -1))
    dX_scan, dY_scan = d_scan[:, np.sum(np.abs(d_scan), axis=0) <= 4]

    # improve gradient for point refining (size is insignificant)
    pdetector += cv2.GaussianBlur(pdetector, (25, 25), 0)
    ndetector += cv2.GaussianBlur(ndetector, (25, 25), 0)

    # iteratively increase chessboard size, start!
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
    points_new = np.dot(coeffs_new, offset) + center
    pthreshold = nthreshold = None
    while True:

        # refine points
        i = (coeffs_new[0, 0] + coeffs_new[0, 1]) % 2
        pslice = np.arange(i, points_new.shape[0], 2)
        nslice = np.arange(1 - i, points_new.shape[0], 2)
        for _ in range(2):  # 2 is empirical value
            X_scan = np.clip(np.add.outer(dX_scan,
                                          np.round(points_new[:, 0]).astype(int)), 0, gray.shape[1] - 1)
            Y_scan = np.clip(np.add.outer(dY_scan,
                                          np.round(points_new[:, 1]).astype(int)), 0, gray.shape[0] - 1)
            pscores = pdetector[Y_scan, X_scan]
            nscores = ndetector[Y_scan, X_scan]
            pindex = pscores.argmax(axis=0)[pslice]
            nindex = nscores.argmax(axis=0)[nslice]
            points_new[pslice, 0] = X_scan[pindex, pslice]
            points_new[pslice, 1] = Y_scan[pindex, pslice]
            points_new[nslice, 0] = X_scan[nindex, nslice]
            points_new[nslice, 1] = Y_scan[nindex, nslice]

        # validate new points
        pscore = np.median(pscores[pindex, pslice])
        if pthreshold is None:
            pthreshold = pscore
        nscore = np.median(nscores[nindex, nslice])
        if nthreshold is None:
            nthreshold = nscore
        if (pscore * 3 < pthreshold or
                nscore * 3 < nthreshold):  # 3 is empirical value
            if action is None:
                raise ValueError  # fail
            actions.remove(action)
            if not actions:
                break
        else:
            pthreshold = min(pthreshold, pscore)
            nthreshold = min(nthreshold, nscore)

            # update homography transformation
            coeffs_new = coeffs_new.astype(np.float32)
            points_new = points_new.astype(np.float32)
            if action is None:
                coeffs = coeffs_new
                points = points_new
            else:
                coeffs = np.append(coeffs, coeffs_new, axis=0)
                points = np.append(points, points_new, axis=0)
            umin, vmin, umax, vmax = bound_new
            if umax - umin > ncol or vmax - vmin > nrow:
                raise ValueError
            mat = cv2.findHomography(coeffs, points)[0]

        # increase chessboard size
        action = actions[0 if action not in actions else
                         actions.index(action) - 1]
        bound_new, coeffs_new = action(umin, vmin, umax, vmax)
        points_new = cv2.perspectiveTransform(
            coeffs_new[None, :, :].astype(np.float32), mat)[0]

    # return top-left, top-right, bottom-left, bottom-right position
    if umax - umin != ncol:
        raise ValueError
    if vmax - vmin != nrow:
        raise ValueError
    return cv2.perspectiveTransform(np.array(
        [[[umin, vmin], [umax, vmin], [umin, vmax], [umax, vmax]]],
        np.float32), mat)[0]


def fit_chessboard_corners(gray, bound, nrow=58, ncol=58, nsample=60):
    if nsample % 2 != 0:
        raise ValueError
    mtx = cv2.getPerspectiveTransform(
        np.reshape(np.meshgrid([0.5 * nsample, (ncol + 0.5) * nsample],
                               [0.5 * nsample, (nrow + 0.5) * nsample]),
                   (2, -1)).T.astype(np.float32, copy=True), bound)
    resample = cv2.warpPerspective(gray, np.linalg.inv(mtx),
                                   ((ncol + 1) * nsample,
                                    (nrow + 1) * nsample))
    corners = np.reshape(np.meshgrid((np.arange(0, ncol + 1) + 0.5) * nsample,
                                     (np.arange(0, nrow + 1) + 0.5) * nsample),
                         (2, -1)).T.astype(np.float32).copy()
    cv2.cornerSubPix(resample, corners,
                     (nsample // 2, nsample // 2),
                     (nsample // 2, nsample // 2),
                     (cv2.TERM_CRITERIA_MAX_ITER, 1, 1))
    return cv2.perspectiveTransform(corners[None, :, :], mtx)[0]


nrow = 58
ncol = 58

cell_width = 588.1 / 58000  # m, horizontal length of grid
cell_height = 580.95 / 58000  # m, vertical length of grid


print('Loading files...')

image_size = None
filenames = []
image_points = []
for root, dirs, files in os.walk(u'raw'):
    for name in files:
        if not name.endswith('.jpg'):
            continue
        filename = os.path.join(root, name)
        image = cv2_imread(filename)
        h, w = image.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            print(filename + ' <image size not match>')
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rotation = board_rotation.get(filename.replace('\\', '/'), '^').upper()
        try:
            if rotation == '^':
                chessboard_bound_2d = find_large_chessboard(gray)
            elif rotation == '>':
                chessboard_bound_2d = cv2.perspectiveTransform(
                    find_large_chessboard(gray.T[::-1])[None, :, :],
                    np.array([[0, -1, w - 1],
                              [1, 0, 0],
                              [0, 0, 1]], dtype=np.float32))[0]
            elif rotation == '<':
                chessboard_bound_2d = cv2.perspectiveTransform(
                    find_large_chessboard(gray[::-1].T)[None, :, :],
                    np.array([[0, 1, 0],
                              [-1, 0, h - 1],
                              [0, 0, 1]], dtype=np.float32))[0]
            elif rotation == 'V':
                chessboard_bound_2d = cv2.perspectiveTransform(
                    find_large_chessboard(gray[::-1, ::-1])[None, :, :],
                    np.array([[-1,  0, w - 1],
                              [0, -1, h - 1],
                              [0,  0,   1]], dtype=np.float32))[0]
            else:
                raise ValueError('invalid rotation {!r}'.format(rotation))
            chessboard_corners_2d = fit_chessboard_corners(
                gray, chessboard_bound_2d)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            raise
            print(filename + ' <chessboard not match>')
            continue
        print(filename)
        filenames.append(filename)
        image_points.append(chessboard_corners_2d)

chessboard_corners_3d = np.hstack([np.reshape(np.meshgrid(
    np.linspace(-ncol, ncol, ncol + 1) * cell_width / 2,
    np.linspace(-nrow, nrow, nrow + 1) * cell_height / 2),
    (2, -1)).T, np.zeros(((ncol + 1) * (nrow + 1), 1))]).astype(np.float32)
object_points = [chessboard_corners_3d] * len(image_points)

print('Calibrating 相机视频demo...')
calibrate_flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
rmse, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size,
    cameraMatrix=None, distCoeffs=None, flags=calibrate_flags)
print('RMSE : {:g}'.format(rmse))
print('size : {:d}, {:d}'.format(*image_size))
print('  fx : {:g}'.format(mtx[0, 0]))
print('  fy : {:g}'.format(mtx[1, 1]))
print('  cx : {:g}'.format(mtx[0, 2]))
print('  cy : {:g}'.format(mtx[1, 2]))
for name, value in zip('k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4'.split(),
                       dist.ravel()):
    print('{:>4} : {:g}'.format(name, value))

mtx2 = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, alpha=0,
                                     centerPrincipalPoint=True)[0]
mtx2[[0, 1], [0, 1]] = np.mean(mtx2[[0, 1], [0, 1]])
with open('camera.py', 'wb') as fout:
    fout.write(b'import numpy as np\n')
    fout.write(b'\n')
    fout.write(b'rmse = %r\n' % rmse)
    fout.write(b'distortion_coefficients = np.array(%r, dtype=np.float32)\n'
               % dist.tolist())
    fout.write(b'raw_camera_matrix = np.array(%r, dtype=np.float32)\n'
               % mtx.tolist())
    fout.write(b'undistort_camera_matrix = np.array(%r, dtype=np.float32)\n'
               % mtx2.tolist())
    fout.write(b'image_size = %d, %d\n' % image_size)

print('Running leave-one-out cross-validation...')
print('   ALL   |   LOO   |   RAW   |   File')
print('--------:|--------:|--------:|:---------')
scale = chessboard_corners_3d.shape[0]**0.5
rmses = [np.linalg.norm(corners - cv2.projectPoints(chessboard_corners_3d,
                                                    rvec, tvec, mtx, dist)[0][:, 0, :]) / scale
         for rvec, tvec, corners in zip(rvecs, tvecs, image_points)]
for rmse, filename in sorted(zip(rmses, filenames), reverse=True):
    i = filenames.index(filename)

    _, mtx2, dist2, _, _ = cv2.calibrateCamera(
        object_points[:i] + object_points[i + 1:],
        image_points[:i] + image_points[i + 1:],
        image_size, cameraMatrix=None, distCoeffs=None,
        flags=calibrate_flags)
    _, rvec, tvec = cv2.solvePnP(
        chessboard_corners_3d, image_points[i], mtx2, dist2)
    loocv = np.linalg.norm(image_points[i] -
                           cv2.projectPoints(chessboard_corners_3d, rvec, tvec,
                                             mtx2, dist2)[0][:, 0, :]) / scale

    raw = np.linalg.norm(image_points[i] -
                         cv2.perspectiveTransform(chessboard_corners_3d[None, :, :2],
                                                  cv2.findHomography(chessboard_corners_3d[:, :2],
                                                                     image_points[i])[0])[0]) / scale

    print(u'{:8.3f} |{:8.3f} |{:8.3f} | {!s}'
          .format(rmse, loocv, raw, filename))
