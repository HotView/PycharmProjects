#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: eph

from __future__ import division

import os
import sys
import json
import argparse
from ast import literal_eval
from itertools import chain
from collections import OrderedDict, defaultdict, deque
from operator import itemgetter
from copy import deepcopy
from shutil import copyfile

import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import pyglet

from .l3dr_config import *


__version__ = '1.52.180621'


# fix Chinese path
cv2_imread = lambda path: cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
cv2_imwrite = lambda path, image: cv2.imencode(os.path.splitext(path)[1],
                                               image)[1].tofile(path)

image_exts = set('.jpg .jpeg .png .tiff .bmp'.split())


class ProgressBar(object):

    def __init__(self, items, prefix='', length=40, stream=sys.stderr):
        self.total = len(items) if hasattr(items, '__len__') else int(items)
        self.prefix = prefix
        self.length = length
        self.stream = stream

    def __enter__(self):
        self.current = 0
        return self.progress

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None: self.progress()
        if self.current > 0: self.stream.write('\n')

    def progress(self):
        if self.current > self.total: return
        if self.current > 0: self.stream.write('\r')
        filled = self.length * self.current // self.total
        self.stream.write('{!s}{!s}  {:d} / {:d}\r{!s}{!s}'.format(
                  self.prefix, '-' * self.length, self.current, self.total,
                  self.prefix, '|' * filled))
        self.current += 1


TINY_NUM = {
        '0': (np.array([0, 1, 2, 0, 2, 0, 2, 0, 2, 0, 1, 2]),
              np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4])),
        '1': (np.array([1, 0, 1, 1, 1, 0, 1, 2]),
              np.array([0, 1, 1, 2, 3, 4, 4, 4])),
        '2': (np.array([0, 1, 2, 2, 0, 1, 2, 0, 0, 1, 2]),
              np.array([0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4])),
        '3': (np.array([0, 1, 2, 2, 1, 2, 2, 0, 1, 2]),
              np.array([0, 0, 0, 1, 2, 2, 3, 4, 4, 4])),
        '4': (np.array([0, 2, 0, 2, 0, 1, 2, 2, 2]),
              np.array([0, 0, 1, 1, 2, 2, 2, 3, 4])),
        '5': (np.array([0, 1, 2, 0, 0, 1, 2, 2, 0, 1, 2]),
              np.array([0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4])),
        '6': (np.array([0, 1, 2, 0, 0, 1, 2, 0, 2, 0, 1, 2]),
              np.array([0, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 4])),
        '7': (np.array([0, 1, 2, 2, 2, 2, 2]),
              np.array([0, 0, 0, 1, 2, 3, 4])),
        '8': (np.array([0, 1, 2, 0, 2, 0, 1, 2, 0, 2, 0, 1, 2]),
              np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4])),
        '9': (np.array([0, 1, 2, 0, 2, 0, 1, 2, 2, 0, 1, 2]),
              np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4])),
        '+': (np.array([1, 0, 1, 2, 1]),
              np.array([1, 2, 2, 2, 3])),
        '-': (np.array([0, 1, 2]),
              np.array([2, 2, 2])),
        '.': (np.array([1]),
              np.array([4])),
        'a': (np.array([0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 0, 2]),
              np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4])),
        'e': (np.array([0, 1, 2, 0, 0, 1, 0, 0, 1, 2]),
              np.array([0, 0, 0, 1, 2, 2, 3, 4, 4, 4])),
        'f': (np.array([0, 1, 2, 0, 0, 1, 0, 0]),
              np.array([0, 0, 0, 1, 2, 2, 3, 4])),
        'i': (np.array([0, 1, 2, 1, 1, 1, 0, 1, 2]),
              np.array([0, 0, 0, 1, 2, 3, 4, 4, 4])),
        'n': (np.array([0, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2]),
              np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])),
        '_': (np.array([0, 1, 2]),
              np.array([4, 4, 4]))}

def draw_tiny_number(image, text, point, color=None, bgcolor=None,
                     halign='center', valign='center'):
    w = 1 + 4 * len(text)
    h = 7
    x, y = map(int, map(round, point))
    x = ( x - 1 if halign == 'left' else
          x - w + 1 if halign == 'right' else
          x - w // 2 )
    y = ( y - 1 if valign == 'top' else
          y - h + 1 if valign == 'bottom' else
          y - h // 2 )
    if bgcolor:
        image[max(0,y):max(0,y+h),max(0,x):max(0,x+w)] = bgcolor
    else:
        image[max(0,y):max(0,y+h),max(0,x):max(0,x+w)] >>= 2
    x += 1
    y += 1

    h, w = image.shape[:2]
    for c in text:
        if c in TINY_NUM:
            X, Y = TINY_NUM[c]
            X = X + x
            Y = Y + y
            sel = (0 <= X) & (X < w) & (0 <= Y) & (Y < h)
            if color:
                image[Y[sel], X[sel]] = color
            else:
                image[Y[sel], X[sel]] |= 0xC0
        x += 4


def load_camera_json(filename):
    with open(filename, 'r') as fin:
        data = json.load(fin)

    image_size = int(data['width']), int(data['height'])
    camera_matrix = np.array(
            [[data['fx'],          0, data['cx']],
             [         0, data['fy'], data['cy']],
             [         0,          0,          1]],
            dtype=np.float32)
    dist_coeffs = np.array(
            [[data['k1'], data['k2'], data['p1'], data['p2'],
              data['k3'], data['k4'], data['k5'], data['k6'],
              data['s1'], data['s2'], data['s3'], data['s4']]],
            dtype=np.float32)
    return image_size, camera_matrix, dist_coeffs


def load_points_ply(filename):
    with open(filename, 'rb') as fin:
        line = fin.readline().rstrip()
        if line != b'ply': raise ValueError

        while not line.startswith(b'format '):
            line = fin.readline().rstrip()
            if not line: raise ValueError
        ply_format = line[7:]

        while not line.startswith(b'element vertex '):
            line = fin.readline().rstrip()
            if not line: raise ValueError
        num_points = int(line[15:])

        keys = []
        dtypes = []
        while True:
            line = fin.readline().rstrip()
            if not line: raise ValueError
            if not line.startswith(b'property '): break
            dtype, key = line[9:].split(b' ', 1)
            keys.append(str(key.decode()))
            dtypes.append(str(dtype.decode()))

        while line != b'end_header':
            line = fin.readline().rstrip()
            if not line: raise ValueError

        if ply_format == b'ascii 1.0':
            dtype = [ (key, dtype.replace('char', 'int8'))
                      for key, dtype in zip(keys, dtypes) ]
            data = np.genfromtxt(fin, dtype=dtype, max_rows=num_points)
        elif ply_format == b'binary_little_endian 1.0':
            dtype = [ (key, np.dtype(dtype.replace('char', 'int8'))
                              .str.replace('>', '<'))
                      for key, dtype in zip(keys, dtypes) ]
            data = np.fromfile(fin, dtype=dtype, count=num_points)
        elif ply_format == b'binary_big_endian 1.0':
            dtype = [ (key, np.dtype(dtype.replace('char', 'int8'))
                              .str.replace('<', '>'))
                      for key, dtype in zip(keys, dtypes) ]
            data = np.fromfile(fin, dtype=dtype, count=num_points)
        else:
            raise NotImplementedError('unknown format {!r}'.format(ply_format))

    return data


def save_points_ply(filename, data, ascii=False):
    with open(filename, 'wb') as fout:
        fout.write(b'ply\n')
        fout.write(b'format ')
        fout.write( b'ascii 1.0' if ascii else
                    b'binary_%s_endian 1.0' % sys.byteorder.encode() )
        fout.write(b'\n')
        fout.write(b'element vertex %d\n' % data.shape[0])
        for key in data.dtype.names:
            dtype = data.dtype.fields[key][0].name.replace('int8', 'char')
            fout.write(b'property %s %s\n' % (dtype.encode(), key.encode()))
        fout.write(b'end_header\n')
        if ascii:
            np.savetxt(fout, data, fmt='%g')
        else:
            data.tofile(fout)


def clip_points(points, xmin=-2, xmax=2, ymin=-2, ymax=2, zmin=-2, zmax=2):
    X = points['x']
    Y = points['y']
    Z = points['z']
    return points[(xmin < X) & (X < xmax) &
                  (ymin < Y) & (Y < ymax) &
                  (zmin < Z) & (Z < zmax)]


def render_points(points, image_size, focal_length, rotation, center,
                  nscale=8, bgcolor=0, znear=0.001):
    BGR = np.array([points['blue'], points['green'], points['red']],
                   dtype=np.uint8).T
    X, Y, Z = np.dot(rotation, [points['x'] - center[0],
                                points['y'] - center[1],
                                points['z'] - center[2]])
    order = np.argsort(Z)[-1:-1-np.sum(Z > znear):-1]
    Z = Z[order]
    Y = Y[order] / Z
    X = X[order] / Z
    BGR = BGR[order]

    image = np.full((1, 1, 3), bgcolor, dtype=np.uint8)
    for i in reversed(range(nscale)):
        w = image_size[0] >> i
        h = image_size[1] >> i
        cx = (w - 1) / 2
        cy = (h - 1) / 2
        fx = fy = focal_length / (1 << i)
        image = cv2.resize(image, (w, h))
        image[np.clip(np.round(Y * fy + cy).astype(int), 0, h - 1),
              np.clip(np.round(X * fx + cx).astype(int), 0, w - 1)] = BGR
    return image


def load_mesh_obj(filename):
    V = []
    T = []
    N = []
    F = []
    mtllib = None
    usemtl = None
    triple = lambda v, t=0, n=0: [int(v), int(t), int(n)]
    with open(filename) as fin:
        for line in fin:
            if line.startswith('#'): continue
            key, _, values = line.partition(' ')
            if key == 'v':
                V.append(values)
            elif key == 'vt':
                T.append(values)
            elif key == 'vn':
                N.append(values)
            elif key == 'f':
                face = []
                for i, value in enumerate(values.split()):
                    if i <= 2:
                        face.append(triple(*value.split('/')))
                    else:
                        face = [face[0], face[2], triple(*value.split('/'))]
                    if i >= 2:
                        F.append(face)
            elif key == 'mtllib':
                mtllib = os.path.join(os.path.dirname(
                             os.path.abspath(filename)), values.strip())
            elif key == 'usemtl':
                usemtl = values.strip()
    V = np.loadtxt(V) if V else None
    T = np.loadtxt(T) if T else None
    N = np.loadtxt(N) if N else None
    F = np.array(F) - 1
    return V, T, N, F, mtllib, usemtl


def save_mesh_obj(filename, V, T, N, F, mtllib=None, usemtl=None):
    with open(filename, 'wb') as fout:
        if mtllib:
            fout.write(b'mtllib ')
            fout.write(os.path.basename(mtllib).encode())
            fout.write(b'\n')
        if V is not None:
            for v in V:
                fout.write(b'v %g %g %g\n' % tuple(v))
        if T is not None:
            for t in T:
                fout.write(b'vt %g %g\n' % tuple(t))
        if N is not None:
            for n in N:
                fout.write(b'vn %g %g %g\n' % tuple(n))
        if usemtl:
            fout.write(b'usemtl ')
            fout.write(usemtl.encode())
            fout.write(b'\n')
        if F is not None:
            for f in F + 1:
                fout.write(b'f %d/%d/%d %d/%d/%d %d/%d/%d\n'
                           % tuple(f.ravel()))


def clip_mesh(V, T, N, F, xmin=-2, xmax=2, ymin=-2, ymax=2, zmin=-2, zmax=2):
    X, Y, Z = V.T
    V_sel = ((xmin < X) & (X < xmax) &
             (ymin < Y) & (Y < ymax) &
             (zmin < Z) & (Z < zmax))
    F_sel = np.all(V_sel[F[:,:,0]], axis=1)
    F = F[F_sel]

    V_map = np.full(V.shape[0], -1)
    V = V[V_sel]
    V_map[V_sel] = np.arange(V.shape[0])
    F[:,:,0] = V_map[F[:,:,0]]

    if T is not None:
        T_sel = np.unique(F[:,:,1])
        T_map = np.full(T.shape[0], -1)
        T = T[T_sel]
        T_map[T_sel] = np.arange(T.shape[0])
        F[:,:,1] = T_map[F[:,:,1]]

    if N is not None:
        N_sel = np.unique(F[:,:,2])
        N_map = np.full(N.shape[0], -1)
        N = N[N_sel]
        N_map[N_sel] = np.arange(N.shape[0])
        F[:,:,2] = N_map[F[:,:,2]]

    return V, T, N, F


def get_face_normal(V, F):
    P1 = V[F[:,0,0]]; P2 = V[F[:,1,0]]; P3 = V[F[:,2,0]]
    P = (P1 + P2 + P3) / 3
    N = np.cross(P1 - P2, P1 - P3)
    N /= np.linalg.norm(N, axis=-1)[:,None]
    return P, N


class MeshRenderer(object):

    def __init__(self, max_width=4000, max_height=4000):
        self.meshes = []  # [ (vertex_list, texture) ]
        self.window = pyglet.window.Window(
                width=max_width, height=max_height, visible=False)
        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)

    def add(self, V, T, N, F, mtllib=None, usemtl=None):

        # create OpenGL (pyglet) vertex list
        vertex_list = [('v3f', V[F[:,:,0]].ravel().tolist())]
        if T is not None and len(T) > 0:
            if T.shape[1] == 2:
                vertex_list.append(('t2f', T[F[:,:,1]].ravel().tolist()))
            elif T.shape[1] == 3:
                vertex_list.append(('c3B', T[F[:,:,1]].ravel().tolist()))
            else:
                raise NotImplementedError
        if N is not None and len(N) > 0:
            vertex_list.append(('n3f', N[F[:,:,2]].ravel().tolist()))
        vertex_list = pyglet.graphics.vertex_list(
                          len(vertex_list[0][1]) // 3, *vertex_list)

        # load texture, assuming single texture
        texture = None
        if mtllib:
            with open(mtllib) as fin:
                for line in fin:
                    if line.startswith('map_Kd '):
                        texture = pyglet.image.load(
                                os.path.join(os.path.dirname(
                                    os.path.abspath(mtllib)),
                                line[7:].rstrip())).texture
                        break

        self.meshes.append((vertex_list, texture))

    def render(self, image_size, focal_length, rotation, center,
                     znear=0.1, zfar=10, ortho=False, depth=False,
                     wireframe=False, line_width=1):

        # set camera intrinsic parameters
        w, h = image_size
        f = focal_length
        pyglet.gl.glViewport(0, 0, w, h)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
        if ortho:
            pyglet.gl.glLoadMatrixd((pyglet.gl.gl.c_double*16)(
                2 * f / w,         0,                               0, 0,
                        0, 2 * f / h,                               0, 0,
                        0,         0,              2 / (zfar - znear), 0,
                        0,         0, (znear + zfar) / (znear - zfar), 1))
        else:
            pyglet.gl.glLoadMatrixd((pyglet.gl.gl.c_double*16)(
                2 * f / w,         0,                                 0, 0,
                        0, 2 * f / h,                                 0, 0,
                        0,         0,   (znear + zfar) / (zfar - znear), 1,
                        0,         0, 2 * znear * zfar / (znear - zfar), 0))

        # set camera extrinsic parameters
        M = np.eye(4)
        M[:3,:3] = rotation
        M[:3,3] = -np.dot(rotation, center)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
        pyglet.gl.glLoadMatrixd((pyglet.gl.gl.c_double*16)(*M.T.ravel()))

        # draw mesh (no shading)
        self.window.clear()
        if wireframe:
            pyglet.gl.glPolygonMode(pyglet.gl.GL_FRONT_AND_BACK,
                                    pyglet.gl.GL_LINE)
            pyglet.gl.glLineWidth(line_width)
        else:
            pyglet.gl.glPolygonMode(pyglet.gl.GL_FRONT_AND_BACK,
                                    pyglet.gl.GL_FILL)
        for vertex_list, texture in self.meshes:
            if texture:
                pyglet.gl.glEnable(texture.target)
                pyglet.gl.glBindTexture(texture.target, texture.id)
            vertex_list.draw(pyglet.gl.GL_TRIANGLES)
            if texture:
                pyglet.gl.glDisable(texture.target)

        if depth:
            buffer = (pyglet.gl.gl.c_float * (h * w))()
            pyglet.gl.glReadPixels(0, 0, w, h, pyglet.gl.GL_DEPTH_COMPONENT,
                                   pyglet.gl.GL_FLOAT, buffer)
            if ortho:
                return np.array(buffer).reshape((h, w)) * (zfar-znear) + znear
            else:
                return np.reciprocal(np.array(buffer).reshape((h, w)) *
                                     (1 / zfar - 1 / znear) + 1 / znear)
        else:
            buffer = (pyglet.gl.GLubyte * (h * w * 3))()
            pyglet.gl.glReadPixels(0, 0, w, h, pyglet.gl.GL_BGR,
                                   pyglet.gl.GL_UNSIGNED_BYTE, buffer)
            return np.array(buffer).reshape((h, w, 3))


def find_large_chessboard(gray, nrow=board_num_rows, ncol=board_num_cols):

    # fix underexposure and overexposure
    gray = np.tanh((gray.astype(np.float32) - gray.mean()) / gray.std())
    gray = cv2.dilate(gray, None) + cv2.erode(gray, None)

    # calculate x-corner detector Ixy * Iyx - Ixx * Iyy
    #     reference:
    #         Gustavo Teodoro Laureano, etc. Topological Detection of
    #         Chessboard Pattern for Camera Calibration.
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
                       [ 0,  0,  0,  0,  0,  0,  0],
                       [ 1,  1,  1,  0, -1, -1, -1],
                       [ 2,  2,  1,  0, -1, -2, -2],
                       [ 3,  2,  1,  0, -1, -2, -3]],
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
                  qualityLevel=0.1, minDistance=5, blockSize=3)[:,0,:]
    points2 = cv2.goodFeaturesToTrack(  # empirical configuration
                  ndetector, maxCorners=(nrow + 1) * (ncol + 1),
                  qualityLevel=0.1, minDistance=5, blockSize=3)[:,0,:]

    # pick a start point
    center = points1[np.argmin(np.sum(np.square(points1 -
        np.median(min(points1, points2, key=len), axis=0)), axis=1))]

    # find approximate row and column direction base on Hessian matrix
    #     as gradient change fastest along row and column
    j, i = np.round(center).astype(int)
    V = np.linalg.eig(np.array([[Ixx[i,j], Ixy[i,j]],
                                [Iyx[i,j], Iyy[i,j]]]))[1]
    V = np.dot([[1, 1], [1, -1]], V.T)
    V /= np.hypot(V[:,0], V[:,1])

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
    dX_scan, dY_scan = d_scan[:,np.sum(np.abs(d_scan), axis=0) <= 4]

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
        i = (coeffs_new[0,0] + coeffs_new[0,1]) % 2
        pslice = np.arange(i, points_new.shape[0], 2)
        nslice = np.arange(1-i, points_new.shape[0], 2)
        for _ in range(2):  # 2 is empirical value
            X_scan = np.clip(np.add.outer(dX_scan,
                np.round(points_new[:,0]).astype(int)), 0, gray.shape[1] - 1)
            Y_scan = np.clip(np.add.outer(dY_scan,
                np.round(points_new[:,1]).astype(int)), 0, gray.shape[0] - 1)
            pscores = pdetector[Y_scan,X_scan]
            nscores = ndetector[Y_scan,X_scan]
            pindex = pscores.argmax(axis=0)[pslice]
            nindex = nscores.argmax(axis=0)[nslice]
            points_new[pslice,0] = X_scan[pindex,pslice]
            points_new[pslice,1] = Y_scan[pindex,pslice]
            points_new[nslice,0] = X_scan[nindex,nslice]
            points_new[nslice,1] = Y_scan[nindex,nslice]

        # validate new points
        pscore = np.median(pscores[pindex,pslice])
        if pthreshold is None: pthreshold = pscore
        nscore = np.median(nscores[nindex,nslice])
        if nthreshold is None: nthreshold = nscore
        if (pscore * 3 < pthreshold or
            nscore * 3 < nthreshold):  # 3 is empirical value
            if action is None: raise ValueError  # fail
            actions.remove(action)
            if not actions: break
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
            if umax - umin > ncol or vmax - vmin > nrow: raise ValueError
            mat = cv2.findHomography(coeffs, points)[0]

        # increase chessboard size
        action = actions[ 0 if action not in actions else
                          actions.index(action) - 1 ]
        bound_new, coeffs_new = action(umin, vmin, umax, vmax)
        points_new = cv2.perspectiveTransform(
                coeffs_new[None,:,:].astype(np.float32), mat)[0]

    # return top-left, top-right, bottom-left, bottom-right position
    if umax - umin != ncol: raise ValueError
    if vmax - vmin != nrow: raise ValueError
    return cv2.perspectiveTransform(np.array(
               [[[umin, vmin], [umax, vmin], [umin, vmax], [umax, vmax]]],
               np.float32), mat)[0]


def fit_chessboard_corners(gray, bound,
                           nrow=board_num_rows, ncol=board_num_cols,
                           nsample=60):
    if nsample % 2 != 0: raise ValueError
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
                     (nsample // 3, nsample // 3),
                     (nsample // 3, nsample // 3),
                     (cv2.TERM_CRITERIA_MAX_ITER, 30, 1))
    return cv2.perspectiveTransform(corners[None,:,:], mtx)[0]


def remove_far_points(points, xmin=-2, xmax=2,
                              ymin=-2, ymax=2,
                              zmin=-2, zmax=2):
    X, Y, Z = points.T
    return points[(xmin < X) & (X < xmax) &
                  (ymin < Y) & (Y < ymax) &
                  (zmin < Z) & (Z < zmax)]


def remove_board_points(points,
                        xmin=(board_num_cols // 2 + 1) * -board_cell_width,
                        xmax=(board_num_cols // 2 + 1) * board_cell_width,
                        ymin=(board_num_rows // 2 + 1) * -board_cell_height,
                        ymax=(board_num_rows // 2 + 1) * board_cell_height,
                        zmin=-0.1, zmax=0.2):
    X, Y, Z = points.T
    return points[~((xmin < X) & (X < xmax) &
                    (ymin < Y) & (Y < ymax) &
                    (zmin < Z) & (Z < zmax))]


def align_groups(points_list, reso=0.001, nscale=4):
    kdtrees = [ cKDTree(points) for points in points_list ]

    # modified ICP algorithm
    with ProgressBar(nscale, 'Aligning groups'.ljust(24)) as progress:
        K = np.zeros(6 * (len(points_list) - 1))
        for iscale in reversed(range(nscale)):
            progress()

            std = reso * (1<<iscale)
            points_sample = [ points[::1<<iscale] for points in points_list ]
            scale = sum(map(len, points_sample)) * (len(points_sample) - 1)

            def func(K):
                Rs = [np.eye(3)]
                Ts = [np.zeros(3)]
                for i in range(len(points_list) - 1):
                    Rs.append(cv2.Rodrigues(K[6*i:6*i+3])[0])
                    Ts.append(K[6*i+3:6*i+6])

                score = 0
                for i, points in enumerate(points_sample):
                    for j, kdtree in enumerate(kdtrees):
                        if i == j: continue
                        score -= np.sum(np.exp(np.square(kdtree.query(
                                     np.dot(points, np.dot(Rs[i].T, Rs[j])) +
                                     np.dot(Ts[i] - Ts[j], Rs[j]),
                                     distance_upper_bound=std * 4
                                     )[0]) / (-2 * std**2)))
                return score / scale

            res = minimize(func, K, method='SLSQP',
                           options=dict(maxiter=100000))
            assert res.success
            K = res.x

    Rs = [np.eye(3)]
    Ts = [np.zeros(3)]
    for i in range(len(points_list) - 1):
        Rs.append(cv2.Rodrigues(K[6*i:6*i+3])[0])
        Ts.append(K[6*i+3:6*i+6])
    points_list = [ np.dot(points, R.T) + T
                    for points, R, T in zip(points_list, Rs, Ts) ]
    return points_list, Rs, Ts


def find_symmetry_plane(points_list, reso=0.001, nscale=7):
    kdtrees = [ cKDTree(points) for points in points_list ]

    # estimate symmetry plane base on modified ICP algorithm
    with ProgressBar(nscale, 'Finding symmetry plane'.ljust(24)) as progress:
        K = np.zeros(3)
        for iscale in reversed(range(nscale)):
            progress()

            std = reso * (1<<iscale)
            points_sample = [ points[::1<<iscale] for points in points_list ]
            scale = sum(map(len, points_sample))

            def func(K):
                b, c, d = K
                n = np.array([1., b, c])
                s = 2 / np.dot(n, n)
                R = np.eye(3) - np.outer(n, n) * s
                T = n * (-d * s)
                score = 0
                for kdtree, points in zip(kdtrees, points_sample):
                    score -= np.sum(np.exp(np.square(kdtree.query(
                                 np.dot(points, R) + T,
                                 distance_upper_bound=std * 4
                                 )[0]) / (-2 * std**2)))
                return score / scale

            res = minimize(func, K, method='SLSQP',
                           options=dict(maxiter=100000))
            assert res.success
            K = res.x

    b, c, d = K
    return np.array([1, b, c, d]) / np.sqrt(1 + b * b + c * c)


def find_hinge_axis(points_list, plane, reso=0.001, nscale=7, hboard=0.4):
    a, b, c, d = plane
    if a < 0: a, b, c, d = -plane
    yaxis = np.array([a, b, c])
    zaxis = np.array([b, -a, 0]) / np.hypot(a, b)
    xaxis = np.cross(yaxis, zaxis)
    rotation = np.array([xaxis, yaxis, zaxis])
    center = np.array([-d / a, 0, 0])

    points_list = [ np.dot(points - center, rotation.T)
                    for points in points_list ]
    for points in points_list: points[:,1] = np.abs(points[:,1])
    kdtrees = [ cKDTree(points) for points in points_list ]

    # estimate hinge axis base on modified ICP algorithm
    with ProgressBar(nscale, 'Finding hinge axis'.ljust(24)) as progress:
        K = np.array([hboard + 1, 1] + [0] * (len(points_list) - 1))
        for iscale in reversed(range(nscale)):
            progress()

            std = reso * (1<<iscale)
            points_sample = [ points[np.dot(points, [K[1] - hboard, 0, K[0]])
                                     > K[0] * hboard][::1<<iscale]
                              for points in points_list ]
            scale = sum(map(len, points_sample)) * (len(points_sample) - 1)

            def func(K):
                angles = np.insert(K[2:], 0, 0.)
                score = 0
                for i, points in enumerate(points_sample):
                    for j, kdtree in enumerate(kdtrees):
                        if i == j: continue
                        R = cv2.Rodrigues(np.array(
                                [0., angles[i] - angles[j], 0.]))[0]
                        T = np.dot([-K[0], 0, K[1]], np.eye(3) - R)
                        score -= np.sum(np.exp(np.square(kdtree.query(
                                     np.dot(points, R) + T,
                                     distance_upper_bound=std * 4
                                     )[0]) / (-2 * std**2)))
                return score / scale

            res = minimize(func, K, method='SLSQP',
                           options=dict(maxiter=100000))
            assert res.success
            K = res.x

    hinge = center - xaxis * K[0] + zaxis * K[1]
    return hinge, K[2:]


def estimate_open_angle(body_points, body_normals, gate_points, gate_normals,
                        areso=np.radians(0.005), nscale=4, nnear=1,
                        tol=0.01, atol=np.radians(10), angle0=np.radians(45)):
    kdtree = cKDTree(np.hstack([body_points / tol, body_normals / -atol]))

    with ProgressBar(nscale * 19,
                     'Estimating open angle'.ljust(24)) as progress:

        def func(a):
            progress()
            R = cv2.Rodrigues(np.array([0, -a, 0]))[0]
            D = kdtree.query(np.hstack([np.dot(gate_points, R / tol),
                                        np.dot(gate_normals, R / atol)]),
                             nnear, distance_upper_bound=4)[0]
            return np.sum(np.exp(-np.square(D)))

        a = angle0
        for iscale in reversed(range(nscale)):
            a = max(np.arange(-9, 10) * (areso * 10**iscale) + a, key=func)

    return a


def refine_camera(image, points, image_size, focal_length, rotation, center):
    BGR = np.array([points['blue'], points['green'], points['red']],
                   dtype=np.uint8).T
    w, h = image_size
    cx = (w - 1) / 2
    cy = (h - 1) / 2
    fx = fy = focal_length

    def func(K):
        R = np.dot(rotation, cv2.Rodrigues(K[:3])[0])
        C = np.add(center, K[3:])
        X, Y, Z = np.dot(R, [points['x'] - C[0],
                             points['y'] - C[1],
                             points['z'] - C[2]])
        Y = np.round(Y / Z * fy + cy).astype(int)
        X = np.round(X / Z * fx + cx).astype(int)
        sel = (X >= 0) & (X < w) & (Y >= 0) & (Y < h) & (Z > 0)
        if np.sum(sel) == 0: return 2
        PCC = np.corrcoef(image[Y[sel], X[sel]].ravel(),
                          BGR[sel].ravel())[0,1]
        if np.isnan(PCC): return 2
        return 1 - PCC

    res = minimize(func, np.zeros(6), method='BFGS',
                   options=dict(maxiter=100000))
    if not res.success or res.fun <= func(np.zeros(6)): return rotation, center
    K = res.x
    R = np.dot(rotation, cv2.Rodrigues(K[:3])[0])
    C = np.add(center, K[3:])
    return R.tolist(), C.tolist()


def mask_from_depth(depth, principal_point, focal_length, rotation, center,
                    xmin=-np.inf, xmax=np.inf,
                    ymin=-np.inf, ymax=np.inf,
                    zmin=-np.inf, zmax=np.inf):
    h, w = depth.shape
    Y, X = np.mgrid[:h,:w]
    Z = depth
    X = (X - principal_point[0]) * Z / focal_length
    Y = (Y - principal_point[1]) * Z / focal_length
    X, Y, Z = (np.dot(np.transpose([X, Y, Z]), rotation) + center).T
    mask = np.ones(depth.shape, dtype=bool)
    if xmin > -np.inf: mask &= (X > xmin)
    if xmax <  np.inf: mask &= (X < xmax)
    if ymin > -np.inf: mask &= (Y > ymin)
    if ymax <  np.inf: mask &= (Y < ymax)
    if zmin > -np.inf: mask &= (Z > zmin)
    if zmax <  np.inf: mask &= (Z < zmax)
    return mask


def init_fisheye_map(image_size, principal_point, focal_length,
                     fisheye_image_size, fisheye_principal_point,
                     fisheye_focal_length):
    w, h = fisheye_image_size
    Y, X = np.mgrid[:h,:w]
    X = X - fisheye_principal_point[0]
    Y = Y - fisheye_principal_point[1]
    D = np.tan(np.hypot(Y, X) / fisheye_focal_length) * focal_length
    A = np.arctan2(Y, X)
    X = D * np.cos(A) + principal_point[0]
    Y = D * np.sin(A) + principal_point[1]
    return X.astype(np.float32), Y.astype(np.float32)


def find_circles(image, mask=None, threshold=200, nmax=100,
                 rmin=5, rmax=25, rstd=1, rpan=4):
    if rmin < rpan + 1: raise ValueError

    if mask is not None:
        rmin = max(rmin, int(np.ceil(np.min(mask) - rstd)))
        rmax = min(rmax, int(np.floor(np.max(mask) + rstd)))
    if rmin > rmax: return [], []

    # generate gradient
    Dx = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    Dy = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    Da = np.arctan2(Dy, Dx) * 2
    Ds = np.log1p(np.hypot(Dy, Dx))
    Du = np.sum(np.cos(Da) * Ds, axis=-1)
    Dv = np.sum(np.sin(Da) * Ds, axis=-1)

    # calculate likelihood for each (x, y, r) pair
    #     based on: gradient changes across circle
    def iter_scores():
        queue = deque()
        for radius in range(rmin - rpan, rmax + rpan + 1):
            r = int(np.ceil(radius + 6 + rstd * 4))
            Ky, Kx = np.mgrid[-r:r+1,-r:r+1]
            Ka = np.arctan2(Ky, Kx) * 2
            Ks = np.exp(np.square(np.hypot(Ky, Kx) - radius) /
                        (-2 * rstd**2)) / np.sqrt(radius)
            Ku = np.cos(Ka) * Ks
            Kv = np.sin(Ka) * Ks
            queue.append(cv2.filter2D(Du, cv2.CV_32F, Ku) +
                         cv2.filter2D(Dv, cv2.CV_32F, Kv))
            if len(queue) > rpan * 2:
                yield (radius - rpan, queue[rpan] -
                       (np.fmax(0, queue[0]) + np.fmax(0, queue[rpan*2])))
                queue.popleft()

    # choose best (x, y, r) for each (x, y)
    radiuses = np.zeros(image.shape[:2], dtype=int)
    scores = np.full(image.shape[:2], -np.inf)
    for radius, score in iter_scores():
        sel = (score > scores)
        if mask is not None:
            sel &= (mask > radius - rstd) & (mask < radius + rstd)
        scores[sel] = score[sel]
        radiuses[sel] = radius

    # choose top n circles
    circles = []
    weights = []
    for _ in range(nmax):
        y, x = np.unravel_index(np.argmax(scores), scores.shape)
        score = scores[y,x]
        if score < threshold: break
        r = radiuses[y,x]
        circles.append((x, y, r))
        weights.append(score)
        cv2.circle(scores, (x, y), r, 0, -1)
    return circles, weights


def refine_points(points_3d_list, points_2d_list, weights_list,
                  rotations, centers, principal_point, focal_length,
                  threshold=5, tol=0.002, distance=0.01):

    with ProgressBar(len(points_3d_list)**2,
                     'Refining points'.ljust(24)) as progress:

        # prepare A and b that  A [x, y, z] -> b
        key2id = {}
        id2key = []
        weights = []
        A = []
        b = []
        for i, points_2d in enumerate(points_2d_list):
            progress()
            for j, (x, y) in enumerate(points_2d):
                a = np.dot([[focal_length, 0, principal_point[0] - x],
                            [0, focal_length, principal_point[1] - y]],
                           rotations[i])
                A.extend(a)
                b.extend(np.dot(a, centers[i]))
                key2id[i,j] = len(key2id)
                id2key.append((i, j))
                weights.append(weights_list[i][j])
        A = np.array(A)
        b = np.array(b)
        id2key = np.array(id2key)
        weights = np.array(weights)

        # for each pair of views,
        #     match pair of circles based on closest (approx) 3d distance
        # for each pair of circles (corresponding rows in A and b),
        #     solve position  [x, y, z]  based on  A [x, y, z] -> b
        #     rank likelihood based on
        #         sum( closeness of projection * weight of point
        #              for each point in each view )
        scores = []
        points = []
        observations = []
        for i, points_3d in enumerate(points_3d_list):
            kdtree = cKDTree(points_3d)
            for k, points_3d in enumerate(points_3d_list):
                if i == k: continue
                progress()
                J = kdtree.query(points_3d, distance_upper_bound=distance)[1]
                L = np.arange(len(points_3d))
                sel = (J < kdtree.n)
                for j, l in zip(J[sel], L[sel]):
                    sel = [key2id[i,j] * 2, key2id[i,j] * 2 + 1,
                           key2id[k,l] * 2, key2id[k,l] * 2 + 1]
                    point = np.linalg.lstsq(A[sel], b[sel], rcond=-1)[0]
                    residuals = np.dot(A, point) - b
                    residuals = np.hypot(residuals[0::2],
                                         residuals[1::2]) / focal_length
                    sel = residuals < tol
                    if np.sum(sel) < threshold: continue
                    scores.append(np.sum(np.exp(residuals**2 / (-2 * tol**2))
                                         * weights))
                    points.append(point)
                    observations.append(id2key[sel])
        order = np.argsort(scores)[::-1]
        points = np.array(points)[order]
        observations = np.array(observations, dtype=object)[order]

    with ProgressBar(points, 'Filtering points'.ljust(24)) as progress:

        # leave only point with highest score in each 3d range
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        sel = np.ones(points.shape[0], dtype=bool)
        for i, point in enumerate(points):
            progress()
            if not sel[i]: continue
            sel[distances[i] < distance] = False
        points = points[sel]
        observations = observations[sel]

    return points, observations


def calibrate_main(camerafile, paths, cross_validation=False,
                   nrow=board_num_rows, ncol=board_num_cols,
                   cell_width=board_cell_width, cell_height=board_cell_height):
    image_size = None
    filenames = []
    image_points = []

    print('Loading...')
    print('')
    print('   |   File')
    print('---|----------')
    for root, dirs, files in chain(*map(os.walk, paths)):

        try:
            with open(os.path.join(root, 'rotation.json')) as fin:
                rotations = literal_eval(fin.read())
            if not isinstance(rotations, dict):
                print(' X | ' + root)
                continue
        except IOError:
            rotations = {}

        for name in files:
            if os.path.splitext(name)[1].lower() not in image_exts: continue
            filename = os.path.join(root, name)
            image = cv2_imread(filename)
            h, w = image.shape[:2]
            if image_size is None:
                image_size = (w, h)
            elif image_size != (w, h):
                print(' X | ' + filename)
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rotation = rotations.get(name, '^').upper()
            try:
                if rotation == '^':
                    chessboard_bound_2d = find_large_chessboard(
                                                  gray, nrow, ncol)
                elif rotation == '>':
                    chessboard_bound_2d = cv2.perspectiveTransform(
                            find_large_chessboard(
                                    gray.T[::-1], nrow, ncol)[None,:,:],
                            np.array([[0, -1, w-1],
                                      [1,  0,   0],
                                      [0,  0,   1]], dtype=np.float32))[0]
                elif rotation == '<':
                    chessboard_bound_2d = cv2.perspectiveTransform(
                            find_large_chessboard(
                                    gray[::-1].T, nrow, ncol)[None,:,:],
                            np.array([[ 0, 1,   0],
                                      [-1, 0, h-1],
                                      [ 0, 0,   1]], dtype=np.float32))[0]
                elif rotation == 'V':
                    chessboard_bound_2d = cv2.perspectiveTransform(
                            find_large_chessboard(
                                    gray[::-1,::-1], nrow, ncol)[None,:,:],
                            np.array([[-1,  0, w-1],
                                      [ 0, -1, h-1],
                                      [ 0,  0,   1]], dtype=np.float32))[0]
                else:
                    print(' X | ' + filename)
                    continue
                chessboard_corners_2d = fit_chessboard_corners(
                        gray, chessboard_bound_2d, nrow, ncol)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print(' X | ' + filename)
                continue
            print(' ' + rotation + ' | ' + filename)
            filenames.append(filename)
            image_points.append(chessboard_corners_2d)
    print('')

    if not filenames: raise ValueError('no valid image')

    chessboard_corners_3d = np.hstack([np.reshape(np.meshgrid(
        np.linspace(-ncol, ncol, ncol + 1) * cell_width / 2,
        np.linspace(-nrow, nrow, nrow + 1) * cell_height / 2),
        (2, -1)).T, np.zeros(((ncol + 1) * (nrow + 1), 1))]).astype(np.float32)
    object_points = [chessboard_corners_3d] * len(image_points)

    print('Calibrating camera...')
    calibrate_flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
    rmse, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size,
            cameraMatrix=None, distCoeffs=None, flags=calibrate_flags)
    print('')
    print(' Name | Value')
    print('------|-------')
    print(' RMSE | {:g}'.format(rmse))
    print(' size | {:d}, {:d}'.format(*image_size))
    print('   fx | {:g}'.format(mtx[0,0]))
    print('   fy | {:g}'.format(mtx[1,1]))
    print('   cx | {:g}'.format(mtx[0,2]))
    print('   cy | {:g}'.format(mtx[1,2]))
    data = OrderedDict([('width', image_size[0]), ('height', image_size[1]),
                        ('fx', mtx[0,0]), ('fy', mtx[1,1]),
                        ('cx', mtx[0,2]), ('cy', mtx[1,2])])
    for name, value in zip('k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4'.split(),
                           dist.ravel()):
        data[name] = value
        print('{:>5} | {:g}'.format(name, value))
    print('')

    with open(camerafile, 'w') as fout:
        json.dump(data, fout, indent=4)

    if not cross_validation: return

    print('Running leave-one-out cross-validation...')
    print('')
    print('   ALL   |   LOO   |   RAW   |   File')
    print('---------|---------|---------|----------')
    scale = chessboard_corners_3d.shape[0]**0.5
    rmses = [ np.linalg.norm(corners - cv2.projectPoints(chessboard_corners_3d,
                  rvec, tvec, mtx, dist)[0][:,0,:]) / scale
              for rvec, tvec, corners in zip(rvecs, tvecs, image_points) ]
    for rmse, filename in sorted(zip(rmses, filenames), reverse=True):
        i = filenames.index(filename)

        _, mtx2, dist2, _, _ = cv2.calibrateCamera(
                object_points[:i] + object_points[i+1:],
                image_points[:i] + image_points[i+1:],
                image_size, cameraMatrix=None, distCoeffs=None,
                flags=calibrate_flags)
        _, rvec, tvec = cv2.solvePnP(
                          chessboard_corners_3d, image_points[i], mtx2, dist2)
        loocv = np.linalg.norm(image_points[i] -
                    cv2.projectPoints(chessboard_corners_3d, rvec, tvec,
                                      mtx2, dist2)[0][:,0,:]) / scale

        raw = np.linalg.norm(image_points[i] -
                  cv2.perspectiveTransform(chessboard_corners_3d[None,:,:2],
                      cv2.findHomography(chessboard_corners_3d[:,:2],
                                         image_points[i])[0])[0]) / scale

        print('{:8.3f} |{:8.3f} |{:8.3f} | '.format(rmse, loocv, raw)
              + filename)
    print('')


def import_main(camerafile, paths, stage=10, expand=0, mirror=False,
                nrow=board_num_rows, ncol=board_num_cols,
                cell_width=board_cell_width, cell_height=board_cell_height):
    if stage < 0: raise ValueError
    if expand < 0: raise ValueError

    image_size, camera_matrix, dist_coeffs = load_camera_json(camerafile)
    focal_length = float(min(camera_matrix[0,0], camera_matrix[1,1]))
    new_camera_matrix = np.array(
            [[focal_length,            0, (image_size[0] - 1) / 2],
             [           0, focal_length, (image_size[1] - 1) / 2],
             [           0,            0,                       1]],
            dtype=np.float32)

    chessboard_corners_3d = np.hstack([
            np.reshape(np.meshgrid(
                np.linspace(-ncol, ncol, ncol + 1) * cell_width / 2,
                np.linspace(-nrow, nrow, nrow + 1) * cell_height / 2
                ), (2, -1)).T,
            np.zeros(((ncol + 1) * (nrow + 1), 1))]).astype(np.float32)

    chessboard_mask_3d = np.hstack([
            np.reshape(np.meshgrid(
                np.linspace(-ncol - 4, ncol + 4, 2) * cell_width / 2,
                np.linspace(-nrow - 4, nrow + 4, 2) * cell_height / 2
                ), (2, -1)).T,
            np.zeros((4, 1))]).astype(np.float32)[[0,1,3,2]]

    intrinsics = [{
        'key': 0,
        'value': {
            'polymorphic_id': 2147483649,
            'polymorphic_name': 'pinhole_radial_k3',
            'ptr_wrapper': {
                'id': 2147483656,
                'data': {
                    'width': image_size[0],
                    'height': image_size[1],
                    'focal_length': focal_length,
                    'principal_point': new_camera_matrix[:2,2].tolist(),
                    'disto_k3': [0., 0., 0.]}}}}]

    if not os.path.exists('images/'): os.makedirs('images/')
    for i, path in enumerate(paths, 1):

        filenames = [ os.path.join(path, name)
                      for name in os.listdir(path)
                      if os.path.splitext(name)[1].lower() in image_exts ]
        unimported = []
        views = []
        extrinsics = []

        k = 0
        with ProgressBar(filenames, 'Importing group {:d}'
                                    .format(i).ljust(24)) as progress:
            for j, filename in enumerate(filenames, 1):
                progress()

                image = cv2_imread(filename)
                if image.shape[1::-1] != image_size: raise ValueError

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                image = cv2.undistort(image, camera_matrix, dist_coeffs,
                                      None, new_camera_matrix)
                cv2_imwrite('images/{:d}_{:d}.jpg'.format(i, j), image)

                mask = np.full(image.shape[:2], 255, dtype=np.uint8)
                mask = cv2.undistort(mask, camera_matrix, dist_coeffs,
                                     None, new_camera_matrix)
                mask = cv2.erode(mask, None)
                mask[0,:] = 0; mask[-1,:] = 0; mask[:,0] = 0; mask[:,-1] = 0

                try:
                    chessboard_bound_2d = find_large_chessboard(
                            gray, nrow, ncol)
                    chessboard_corners_2d = fit_chessboard_corners(
                            gray, chessboard_bound_2d, nrow, ncol)
                except ValueError:
                    unimported.append(j)
                    cv2_imwrite('images/{:d}_{:d}_mask.png'.format(i, j), mask)
                    continue

                _, rvec, tvec = cv2.solvePnP(chessboard_corners_3d,
                                             chessboard_corners_2d,
                                             camera_matrix, dist_coeffs)
                chessboard_mask_2d = np.round(cv2.projectPoints(
                        chessboard_mask_3d, rvec, tvec,
                        new_camera_matrix, None)[0][:,0,:]).astype(int)
                cv2.fillPoly(mask, [chessboard_mask_2d], 0)
                cv2_imwrite('images/{:d}_{:d}_mask.png'.format(i, j), mask)

                rotation = cv2.Rodrigues(rvec)[0]
                center = np.dot(rotation.T, -tvec.ravel())
                views.append({
                    'key': k,
                    'value': {
                        'polymorphic_id': 1073741824,
                        'ptr_wrapper': {
                            'id': 2147483649 + k,
                            'data': {
                                'local_path': '',
                                'filename': '{:d}_{:d}.jpg'.format(i, j),
                                'width': image_size[0],
                                'height': image_size[1],
                                'id_view': k,
                                'id_intrinsic': 0,
                                'id_pose': k}}}})
                extrinsics.append({
                    'key': k,
                    'value': {
                        'rotation': rotation.tolist(),
                        'center': center.tolist()}})
                k += 1

        groupname = 'group' + str(i)
        if not os.path.exists(groupname): os.makedirs(groupname)
        if not os.path.exists(os.path.join(groupname, 'unimported_images')):
            os.makedirs(os.path.join(groupname, 'unimported_images'))
        for j in unimported:
            copyfile('images/{:d}_{:d}.jpg'.format(i, j),
                     'group{:d}/unimported_images/{:d}_{:d}.jpg'
                     .format(i, i, j))

        with open(groupname + '/' + groupname + '.json', 'w') as fout:
            json.dump(OrderedDict([
                          ('sfm_data_version', '0.3'),
                          ('root_path', '../images'),
                          ('views', views),
                          ('intrinsics', intrinsics),
                          ('extrinsics', extrinsics),
                          ('structure', []),
                          ('control_points', [])]), fout, indent=4)

    python_path = sys.executable
    script_path = os.path.abspath(__file__)
    workdir = os.path.abspath('.')
    with open('run.bat', 'w') as fout:

        # reconstruct in board coordinates
        groupnames = [ 'group' + str(i + 1) for i in range(len(paths)) ]
        for groupname in groupnames:
            suffix = ''

            fout.write('cd "{!s}"\n'.format(os.path.join(workdir, groupname)))
            fout.write('"{0}" -i {1}.json -o . -p ULTRA\n'.format(
                       os.path.join(
                           openmvg_path, 'openMVG_main_ComputeFeatures'),
                       groupname))

            for i in range(stage + 1):
                if i == 0:
                    new_suffix = suffix + '_stage0'
                else:
                    new_suffix = suffix[:-len(str(i - 1))] + str(i)
                    fout.write('"{0}" "{1}" adjust {2}{3}.json\n'.format(
                               python_path, script_path, groupname, suffix))
                fout.write('"{0}" -i {1}{2}.json -m . -o {1}{3}.json\n'.format(
                           os.path.join(
                               openmvg_path,
                               'openMVG_main_ComputeStructureFromKnownPoses'),
                           groupname, suffix, new_suffix))
                suffix = new_suffix

            for i in range(expand):
                if i == 0:
                    new_suffix = suffix + '_expand1'
                else:
                    new_suffix = suffix[:-len(str(i))] + str(i + 1)
                fout.write('"{0}" -i {1}{2}.json -o {1}{2}.mvs\n'.format(
                           os.path.join(
                               openmvg_path, 'openMVG_main_openMVG2openMVS'),
                           groupname, suffix))
                fout.write('"{0}" -i {1}{2}.mvs -o {1}{2}_dense.mvs\n'.format(
                           os.path.join(openmvs_path, 'DensifyPointCloud'),
                           groupname, suffix))
                fout.write('"{0}" -i {1}{2}.json -m . -o .'
                           ' -q unimported_images -s\n'.format(
                           os.path.join(
                               openmvg_path, 'openMVG_main_SfM_Localization'),
                           groupname, suffix))
                fout.write('"{0}" "{1}" expand sfm_data_expanded.json'
                           ' {2}{3}_dense.ply\n'.format(
                           python_path, script_path, groupname, suffix))
                fout.write('"{0}" -i sfm_data_expanded.json -m . '
                           '-o {1}{2}.json\n'.format(
                           os.path.join(
                               openmvg_path,
                               'openMVG_main_ComputeStructureFromKnownPoses'),
                           groupname, new_suffix))
                suffix = new_suffix

            fout.write('"{0}" -i {1}{2}.json -o {1}{2}.mvs\n'.format(
                       os.path.join(
                           openmvg_path, 'openMVG_main_openMVG2openMVS'),
                       groupname, suffix))
            fout.write('"{0}" -i {1}{2}.mvs -o {1}{2}_dense.mvs\n'.format(
                       os.path.join(openmvs_path, 'DensifyPointCloud'),
                       groupname, suffix))
            fout.write('"{0}" "{1}" render {2}{3}.json'
                       ' {2}{3}_dense.ply\n'.format(
                       python_path, script_path, groupname, suffix))

        # project to hinge coordinates
        fout.write('cd "{!s}"\n'.format(workdir))
        fout.write('"{0}" "{1}" align {2}\n'.format(
            python_path, script_path,
            ' '.join( '"{0}/{0}{1}.json" "{0}/{0}{1}_dense.ply"'
                      .format(groupname, suffix)
                      for groupname in groupnames )))
        groupnames = ['body', 'gate']
        if mirror:
            for groupname in groupnames:
                fout.write('"{0}" "{1}" mirror "{2}/{2}.json" '
                           '"{2}_mirror/{2}_mirror.json"\n'.format(
                           python_path, script_path, groupname))
            groupnames.append('body_mirror')
            groupnames.append('gate_mirror')

        # reconstruct in hinge coordinates
        for groupname in groupnames:
            fout.write('cd "{!s}"\n'.format(os.path.join(workdir, groupname)))
            fout.write('"{0}" -i {1}.json -o {1}.mvs\n'.format(
                       os.path.join(
                           openmvg_path, 'openMVG_main_openMVG2openMVS'),
                       groupname))
            fout.write('"{0}" -i {1}.mvs -o {1}_dense.mvs\n'.format(
                       os.path.join(openmvs_path, 'DensifyPointCloud'),
                       groupname))
            fout.write('"{0}" "{1}" render {2}.json {2}_dense.ply\n'.format(
                       python_path, script_path, groupname))
            fout.write('"{0}" -i {1}_dense.mvs -o {1}_dense_mesh.mvs\n'.format(
                       os.path.join(openmvs_path, 'ReconstructMesh'),
                       groupname))
            fout.write('"{0}" -i {1}_dense_mesh.mvs '
                             '-o {1}_dense_mesh_refine.mvs\n'.format(
                       os.path.join(openmvs_path, 'RefineMesh'), groupname))
            fout.write('"{0}" -i {1}_dense_mesh_refine.mvs '
                             '-o {1}_dense_mesh_refine_texture.mvs '
                             '--export-type obj\n'.format(
                       os.path.join(openmvs_path, 'TextureMesh'), groupname))
            fout.write('"{0}" "{1}" clip -xmin=0 -xmax=2 -ymin=-1 -ymax=1 {3}'
                       ' {2}_dense_mesh_refine_texture.obj'
                       ' {2}_dense_mesh_refine_texture_clip.obj\n'.format(
                       python_path, script_path, groupname,
                       ( '-zmin=0 -zmax=2' if groupname.startswith('gate') else
                         '-zmin=-2 -zmax=0' )))
            fout.write('"{0}" "{1}" render {2}.json'
                       ' {2}_dense_mesh_refine_texture_clip.obj\n'.format(
                       python_path, script_path, groupname))
            if groupname.startswith('body'):
                fout.write('"{0}" "{1}" joint {2}.json'
                           ' {2}_dense_mesh_refine_texture_clip.obj\n'.format(
                           python_path, script_path, groupname))
            else:
                fout.write('"{0}" "{1}" close'
                           ' "../body{2}/body{2}_dense_mesh_refine_texture_clip.obj"'
                           ' gate{2}_dense_mesh_refine_texture_clip.obj'
                           ' gate{2}.json\n'.format(
                           python_path, script_path, groupname[4:]))
                fout.write('"{0}" "{1}" joint {2}_close.json'
                           ' {2}_dense_mesh_refine_texture_clip_close.obj\n'.format(
                           python_path, script_path, groupname))
                fout.write('"{0}" "{1}" strut'
                           ' "../body{2}/body{2}_joint.json"'
                           ' "../body{2}/body{2}_dense_mesh_refine_texture_clip.obj"'
                           ' gate{2}_close_joint.json'
                           ' gate{2}_dense_mesh_refine_texture_clip_close.obj'
                           ' "../strut{2}.png"\n'.format(
                           python_path, script_path, groupname[4:]))


def adjust_main(infile, outfile=None):
    if not outfile: outfile = infile

    with open(infile) as fin:
        data = json.load(fin)

    intrinsic_data = data['intrinsics'][0]['value']['ptr_wrapper']['data']
    camera_matrix = np.eye(3, dtype=np.float32)
    camera_matrix[0,0] = intrinsic_data['focal_length']
    camera_matrix[1,1] = intrinsic_data['focal_length']
    camera_matrix[:2,2] = intrinsic_data['principal_point']

    points_3d = []
    points_2d_list = defaultdict(list)
    points_2d_mask = defaultdict(list)
    for i, point in enumerate(data['structure']):
        points_3d.append(point['value']['X'])
        for observation in point['value']['observations']:
            j = observation['key']
            points_2d_list[j].append(observation['value']['x'])
            points_2d_mask[j].append(i)
    points_3d = np.array(points_3d, dtype=np.float32)

    for extrinsic in data['extrinsics']:
        j = extrinsic['key']
        points_2d = np.array(points_2d_list[j], dtype=np.float32)
        _, rvec, tvec, _ = cv2.solvePnPRansac(
            points_3d[points_2d_mask[j]], points_2d, camera_matrix, None)
        rotation = cv2.Rodrigues(rvec)[0]
        center = np.dot(rotation.T, -tvec.ravel())
        extrinsic['value']['rotation'] = rotation.tolist()
        extrinsic['value']['center'] = center.tolist()

    with open(outfile, 'w') as fout:
        json.dump(OrderedDict([
                      ('sfm_data_version', '0.3'),
                      ('root_path', '../images'),
                      ('views', data['views']),
                      ('intrinsics', data['intrinsics']),
                      ('extrinsics', data['extrinsics']),
                      ('structure', []),
                      ('control_points', [])]), fout, indent=4)


def expand_main(infile, plyfile, outfile=None,
                pcc_threshold=0.4, optimize=False):
    if not outfile: outfile = infile
    inpath = os.path.dirname(infile)

    with open(infile) as fin:
        data = json.load(fin)
    intrinsic_data = data['intrinsics'][0]['value']['ptr_wrapper']['data']
    focal_length = intrinsic_data['focal_length']
    image_size = intrinsic_data['width'], intrinsic_data['height']

    points = load_points_ply(plyfile)

    extrinsics = []
    views = []
    num_validated = 0
    i = 0  # new view/extrinsic id
    get_extrinsic = dict( (extrinsic['key'], extrinsic )
                          for extrinsic in data['extrinsics'] ).get
    with ProgressBar(data['views'], 'Validating'.ljust(24)) as progress:
        for view in data['views']:
            progress()

            extrinsic = get_extrinsic(
                    view['value']['ptr_wrapper']['data']['id_pose'])
            if extrinsic is None: continue

            filename = os.path.join(
                    inpath, 'unimported_images',
                    view['value']['ptr_wrapper']['data']['filename'])
            if os.path.exists(filename):
                raw = cv2_imread(filename)

                # optimize camera position
                if optimize:
                    (extrinsic['value']['rotation'],
                     extrinsic['value']['center']) = refine_camera(
                            raw, points, image_size, focal_length,
                            extrinsic['value']['rotation'],
                            extrinsic['value']['center'])

                # validate camera position
                image = render_points(
                            points, image_size, focal_length,
                            extrinsic['value']['rotation'],
                            extrinsic['value']['center'],
                            bgcolor=int(np.round(np.mean(raw))))
                if np.corrcoef(raw.ravel(),
                               image.ravel())[0,1] < pcc_threshold: continue
                os.remove(filename)
                num_validated += 1

            view['key'] = i
            view['value']['ptr_wrapper']['id'] = 2147483649 + i
            view['value']['ptr_wrapper']['data']['id_view'] = i
            view['value']['ptr_wrapper']['data']['id_pose'] = i
            view['value']['ptr_wrapper']['data']['local_path'] = ''
            views.append(view)
            extrinsic['key'] = i
            extrinsics.append(extrinsic)
            i += 1

    print('Validated: {:d}'.format(num_validated))

    with open(outfile, 'w') as fout:
        json.dump(OrderedDict([
                      ('sfm_data_version', '0.3'),
                      ('root_path', '../images'),
                      ('views', views),
                      ('intrinsics', data['intrinsics']),
                      ('extrinsics', extrinsics),
                      ('structure', []),
                      ('control_points', [])]), fout, indent=4)


def align_main(jsonfiles, plyfiles, clip=False):
    if len(jsonfiles) != len(plyfiles): raise ValueError
    if len(jsonfiles) < 2: raise ValueError

    data_list = []
    with ProgressBar(jsonfiles, 'Loading groups'.ljust(24)) as progress:
        for jsonfile in jsonfiles:
            progress()
            with open(jsonfile) as fin:
                data_list.append(json.load(fin))

    raw_points_list = []
    points_list = []
    with ProgressBar(plyfiles, 'Loading points'.ljust(24)) as progress:
        for plyfile in plyfiles:
            progress()
            points = load_points_ply(plyfile)
            raw_points_list.append(points)
            points = np.array([points['x'], points['y'], points['z']]).T
            points = remove_far_points(points)
            points = remove_board_points(points)
            points_list.append(points)

    points_list, Rs, Ts = align_groups(points_list)

    plane = find_symmetry_plane(points_list)
    print('Found plane: {:g}, {:g}, {:g}, {:g}'.format(*plane))

    hinge, angles = find_hinge_axis(points_list, plane)
    angles = np.insert(angles, 0, 0.)
    print('Found hinge: {:g}, {:g}, {:g}'.format(*hinge))
    print('Angles related to group1: ' +
          ', '.join( '{:g}'.format(angle) for angle in np.degrees(angles) ))

    with open('align.json', 'w') as fout:
        json.dump({'plane': plane.tolist(),
                   'hinge': hinge.tolist(),
                   'angles': angles.tolist()}, fout, indent=4)

    a, b, c, d = plane
    yaxis = np.array([a, b, c])
    zaxis = np.array([b, -a, 0]) / np.hypot(a, b)
    xaxis = np.cross(yaxis, zaxis)
    rotation = np.array([xaxis, yaxis, zaxis])

    with ProgressBar(4, 'Saving'.ljust(24)) as progress:
        for groupname, angles in [('gate', angles),
                                  ('body', np.zeros(angles.shape))]:
            if not os.path.exists(groupname): os.makedirs(groupname)

            progress()

            views = []
            extrinsics = []
            structure = []
            new_points_list = []
            i = 0  # view/extrinsic id
            j = 0  # structure id
            for angle, data, points, R, T in zip(
                    angles, data_list, raw_points_list, Rs, Ts):
                C = np.dot(R.T, hinge - T)
                R = np.dot(np.dot(R.T, rotation.T),
                           cv2.Rodrigues(np.array([0, angle, 0]))[0])

                points = points.copy()
                points['x'], points['y'], points['z'] = np.dot(
                        R.T, [points['x'] - C[0],
                              points['y'] - C[1],
                              points['z'] - C[2]])
                new_points_list.append(points)

                remap = []
                for view, extrinsic in zip(data['views'], data['extrinsics']):
                    remap.append(i)
                    view = deepcopy(view)
                    view['key'] = i
                    view['value']['ptr_wrapper']['id'] = 2147483649 + i
                    view['value']['ptr_wrapper']['data']['id_view'] = i
                    view['value']['ptr_wrapper']['data']['id_pose'] = i
                    views.append(view)
                    extrinsic = deepcopy(extrinsic)
                    extrinsic['key'] = i
                    extrinsic['value']['center'] = np.dot(
                            np.array(extrinsic['value']['center']) - C,
                            R).tolist()
                    extrinsic['value']['rotation'] = np.dot(
                            np.array(extrinsic['value']['rotation']),
                            R).tolist()
                    extrinsics.append(extrinsic)
                    i += 1

                for point in data['structure']:
                    P = np.dot(np.array(point['value']['X']) - C, R)

                    # remove insignificant points
                    if clip:
                        if not -0.5 < P[0] < 1.5: continue
                        if not -1 < P[1] < 1: continue
                        if groupname == 'body':
                            if not -2 < P[2] < 0: continue
                        else:
                            if not 0 < P[2] < 2: continue

                    point = deepcopy(point)
                    point['key'] = j
                    point['value']['X'] = P.tolist()
                    for observation in point['value']['observations']:
                        observation['key'] = remap[observation['key']]
                    structure.append(point)
                    j += 1

            progress()

            points = np.concatenate(new_points_list)

            # remove insignificant points
            if clip:
                if groupname == 'body':
                    points = clip_points(points,
                                         xmin=-0.5, xmax=1.5,
                                         ymin=-1, ymax=1,
                                         zmin=-2, zmax=0)
                else:
                    points = clip_points(points,
                                         xmin=-0.5, xmax=1.5,
                                         ymin=-1, ymax=1,
                                         zmin=0, zmax=2)

            save_points_ply(groupname + '/' + groupname + '.ply', points)

            with open(groupname + '/' + groupname + '.json', 'w') as fout:
                json.dump(OrderedDict([
                              ('sfm_data_version', '0.3'),
                              ('root_path', '../images'),
                              ('views', views),
                              ('intrinsics', data_list[0]['intrinsics']),
                              ('extrinsics', extrinsics),
                              ('structure', structure),
                              ('control_points', [])]), fout, indent=4)


def mirror_main(infile, outfile):

    inpath = os.path.dirname(infile)
    with open(infile) as fin:
        data = json.load(fin)
    image_path = os.path.join(inpath, data['root_path'])
    intrinsic_data = data['intrinsics'][0]['value']['ptr_wrapper']['data']
    image_width = intrinsic_data['width']

    with ProgressBar(data['views'], 'Mirroring'.ljust(24)) as progress:
        views = []
        extrinsics = []
        i = 0  # view/extrinsic id
        remap = []
        for view, extrinsic in zip(data['views'], data['extrinsics']):
            progress()

            remap.append(i)

            filename = view['value']['ptr_wrapper']['data']['filename']
            mirror_filename = 'mirror_' + filename

            # mirror json
            view['key'] = i
            view['value']['ptr_wrapper']['id'] = 2147483649 + i
            view['value']['ptr_wrapper']['data']['id_view'] = i
            view['value']['ptr_wrapper']['data']['id_pose'] = i
            views.append(view)
            extrinsic['key'] = i
            extrinsics.append(extrinsic)
            i += 1

            view = deepcopy(view)
            extrinsic = deepcopy(extrinsic)

            view['key'] = i
            view['value']['ptr_wrapper']['id'] = 2147483649 + i
            view['value']['ptr_wrapper']['data']['id_view'] = i
            view['value']['ptr_wrapper']['data']['id_pose'] = i
            view['value']['ptr_wrapper']['data']['filename'] = mirror_filename
            views.append(view)
            extrinsic['key'] = i
            extrinsic['value']['center'][1] *= -1  # mirror Y axis
            extrinsic['value']['rotation'] = np.dot(np.dot(
                    [[-1,0,0],[0,1,0],[0,0,1]],
                    extrinsic['value']['rotation']),
                    [[1,0,0],[0,-1,0],[0,0,1]]).tolist()
            extrinsics.append(extrinsic)
            i += 1

            # mirror image
            image = cv2_imread(os.path.join(image_path, filename))
            image = image[:,::-1]  # horizontal flip,
                                   # assuming principal point at image center
            cv2_imwrite(os.path.join(image_path, mirror_filename), image)

            # mirror mask
            filename = os.path.splitext(filename)[0] + '_mask.png'
            mirror_filename = 'mirror_' + filename

            image = cv2_imread(os.path.join(image_path, filename))
            image = image[:,::-1]  # horizontal flip,
                                   # assuming principal point at image center
            cv2_imwrite(os.path.join(image_path, mirror_filename), image)

    structure = []
    j = 0  # structure id
    for point in data['structure']:
        point['key'] = j
        for observation in point['value']['observations']:
            observation['key'] = remap[observation['key']]
        structure.append(point)
        j += 1

        point = deepcopy(point)

        point['key'] = j
        point['value']['X'][1] *= -1  # mirror Y axis
        for observation in point['value']['observations']:
            observation['key'] += 1
            observation['value']['x'][0] = image_width - \
                                           observation['value']['x'][0]
        structure.append(point)
        j += 1

    outpath = os.path.dirname(os.path.abspath(outfile))
    if not os.path.exists(outpath): os.makedirs(outpath)
    with open(outfile, 'w') as fout:
        json.dump(OrderedDict([
                      ('sfm_data_version', '0.3'),
                      ('root_path', '../images'),
                      ('views', views),
                      ('intrinsics', data['intrinsics']),
                      ('extrinsics', extrinsics),
                      ('structure', structure),
                      ('control_points', [])]), fout, indent=4)

    try:
        points = load_points_ply(os.path.splitext(infile)[0] + '.ply')
    except:
        pass
    else:
        mirror_points = points.copy()
        mirror_points['y'] *= -1
        save_points_ply(os.path.splitext(outfile)[0] + '.ply',
                        np.concatenate([points, mirror_points]))


def clip_main(infile, outfile,
              xmin=-2, xmax=2, ymin=-2, ymax=2, zmin=-2, zmax=2):
    ext = os.path.splitext(infile)[1].lower()

    if ext == '.ply':
        points = load_points_ply(infile)
        points = clip_points(points, xmin, xmax, ymin, ymax, zmin, zmax)
        save_points_ply(outfile, points)

    elif ext == '.obj':
        V, T, N, F, mtllib, usemtl = load_mesh_obj(infile)
        V, T, N, F = clip_mesh(V, T, N, F, xmin, xmax, ymin, ymax, zmin, zmax)
        save_mesh_obj(outfile, V, T, N, F, mtllib, usemtl)

    else:
        raise NotImplementedError


def close_main(bodyobj, gateobj, gatejson=None):

    with ProgressBar(2, 'Loading'.ljust(24)) as progress:

        progress()
        V, T, N, F, mtllib, usemtl = load_mesh_obj(bodyobj)
        body_points, body_normals = get_face_normal(V, F)

        progress()
        V, T, N, F, mtllib, usemtl = load_mesh_obj(gateobj)
        gate_points, gate_normals = get_face_normal(V, F)

    open_angle = estimate_open_angle(body_points, body_normals,
                                     gate_points, gate_normals)
    print('Open angle: {:g}'.format(np.degrees(open_angle)))

    with open('close.json', 'w') as fout:
        json.dump({'open_angle': float(open_angle)}, fout, indent=4)

    R = cv2.Rodrigues(np.array([0, -open_angle, 0]))[0]
    V = np.dot(V, R)
    if N is not None: N = np.dot(N, R)
    save_mesh_obj(os.path.splitext(gateobj)[0] + '_close.obj',
                  V, T, N, F, mtllib, usemtl)

    if gatejson is None: return

    with open(gatejson) as fin:
        data = json.load(fin)

    for extrinsic in data['extrinsics']:
        extrinsic['value']['center'] = np.dot(extrinsic['value']['center'],
                                              R).tolist()
        extrinsic['value']['rotation'] = np.dot(extrinsic['value']['rotation'],
                                                R).tolist()

    for point in data['structure']:
        point['value']['X'] = np.dot(point['value']['X'], R).tolist()

    with open(os.path.splitext(gatejson)[0] + '_close.json', 'w') as fout:
        json.dump(OrderedDict([
                      ('sfm_data_version', '0.3'),
                      ('root_path', '../images'),
                      ('views', data['views']),
                      ('intrinsics', data['intrinsics']),
                      ('extrinsics', data['extrinsics']),
                      ('structure', data['structure']),
                      ('control_points', [])]), fout, indent=4)


def joint_main(infile, objfile, outfile=None,
               nrender=10, radius=0.005, depth_expand=100, rmin=5, rmax=25,
               xmin=0, xmax=0.6, ymin=0.45, ymax=0.7, zmin=-0.7, zmax=-0.05):
    if not outfile: outfile = os.path.splitext(infile)[0] + '_joint.json'
    ymin, ymax = min(abs(ymin), abs(ymax)), max(abs(ymin), abs(ymax))

    with ProgressBar(3, 'Loading'.ljust(24)) as progress:

        progress()
        with open(infile) as fin:
            data = json.load(fin)
        intrinsic_data = data['intrinsics'][0]['value']['ptr_wrapper']['data']
        focal_length = intrinsic_data['focal_length']
        principal_point = intrinsic_data['principal_point']
        image_size = intrinsic_data['width'], intrinsic_data['height']
        image_path = os.path.join(os.path.dirname(infile), data['root_path'])
        w, h = image_size

        progress()
        V, T, N, F, mtllib, usemtl = load_mesh_obj(objfile)

        progress()
        renderer = MeshRenderer(*image_size)
        renderer.add(V, None, None, F)

    with ProgressBar(data['views'],
                     'Finding ball joints'.ljust(24)) as progress:
        fisheye_focal_length = np.hypot(*image_size) / 2 / np.arctan(
                                   np.hypot(*image_size) / 2 / focal_length)
        fisheye_image_size = (int(np.arctan(image_size[0] / 2 / focal_length)
                                  * fisheye_focal_length * 2),
                              int(np.arctan(image_size[1] / 2 / focal_length)
                                  * fisheye_focal_length * 2))
        fisheye_principal_point = ((fisheye_image_size[0] - 1) / 2,
                                   (fisheye_image_size[1] - 1) / 2)
        fisheye_map = init_fisheye_map(
                image_size, principal_point, focal_length, fisheye_image_size,
                fisheye_principal_point, fisheye_focal_length)

        points_3d_list = []
        points_2d_list = []
        weights_list = []
        rotations = []
        centers = []
        filenames = []
        for view, extrinsic in zip(data['views'], data['extrinsics']):
            progress()

            rotation = extrinsic['value']['rotation']
            center = extrinsic['value']['center']
            filename = view['value']['ptr_wrapper']['data']['filename']
            image = cv2_imread(os.path.join(image_path, filename))
            depth = renderer.render(image_size, focal_length,
                                    rotation, center, depth=True)
            max_depth = np.max(depth)
            if depth_expand > 0:
                sel = (depth < max_depth)
                for _ in range(depth_expand):
                    depth = np.where(sel, depth, cv2.erode(depth, None))
                    sel = cv2.dilate(sel.astype(np.uint8), None).astype(bool)

            # convert to fisheye, so projection of ball is circle
            image2 = cv2.remap(image, *fisheye_map,
                              interpolation=cv2.INTER_CUBIC)
            depth2 = cv2.remap(depth - max_depth, *fisheye_map,
                               interpolation=cv2.INTER_NEAREST) + max_depth
            radiuses = fisheye_focal_length * np.arcsin(radius / depth2 /
                           np.hypot(np.hypot(*map(np.subtract,
                           np.mgrid[:fisheye_image_size[1],
                                    :fisheye_image_size[0]],
                           fisheye_principal_point[::-1]))
                           / fisheye_focal_length, 1))
            weights = []
            circles = []

            # find circles (left)
            mask = mask_from_depth(
                       depth, principal_point, focal_length, rotation, center,
                       xmin=xmin, ymin=-ymax, zmin=zmin,
                       xmax=xmax, ymax=-ymin, zmax=zmax)
            mask = cv2.remap(mask.astype(np.uint8), *fisheye_map,
                             interpolation=cv2.INTER_NEAREST)
            argwhere_mask = np.argwhere(mask > 0)
            if argwhere_mask.shape[0] > 0:
                y0, x0 = np.maximum(0, np.min(argwhere_mask, axis=0) - rmax)
                y1, x1 = np.max(argwhere_mask, axis=0) + (rmax + 1)
                circles_l, weights_l = find_circles(
                        image2[y0:y1,x0:x1],
                        radiuses[y0:y1,x0:x1] * mask[y0:y1,x0:x1],
                        rmin=rmin, rmax=rmax)
                if circles_l:
                    weights.extend(weights_l)
                    circles.extend(np.add(circles_l, [x0, y0, 0]))

            # find circles (right)
            mask = mask_from_depth(
                       depth, principal_point, focal_length, rotation, center,
                       xmin=xmin, ymin=ymin, zmin=zmin,
                       xmax=xmax, ymax=ymax, zmax=zmax)
            mask = cv2.remap(mask.astype(np.uint8), *fisheye_map,
                             interpolation=cv2.INTER_NEAREST)
            argwhere_mask = np.argwhere(mask > 0)
            if argwhere_mask.shape[0] > 0:
                y0, x0 = np.maximum(0, np.min(argwhere_mask, axis=0) - rmax)
                y1, x1 = np.max(argwhere_mask, axis=0) + (rmax + 1)
                circles_r, weights_r = find_circles(
                        image2[y0:y1,x0:x1],
                        radiuses[y0:y1,x0:x1] * mask[y0:y1,x0:x1],
                        rmin=rmin, rmax=rmax)
                if circles_r:
                    weights.extend(weights_r)
                    circles.extend(np.add(circles_r, [x0, y0, 0]))

            # convert image circle (xi, yi, ri) to world ball (xo, yo, zo, ro)
            if circles:
                X, Y, R = np.transpose(circles)
                Z = depth2[Y,X]
                X = X - fisheye_principal_point[0]
                Y = Y - fisheye_principal_point[1]
                D = np.tan(np.hypot(Y, X) / fisheye_focal_length)
                A = np.arctan2(Y, X)
                X = D * np.cos(A)
                Y = D * np.sin(A)
                points_3d = np.dot(np.transpose([X * Z, Y * Z, Z]),
                                   rotation) + center
                points_2d = np.transpose([
                                X * focal_length + principal_point[0],
                                Y * focal_length + principal_point[1]])
            else:
                points_3d = np.zeros((0, 3))
                points_2d = np.zeros((0, 2))

            points_3d_list.append(points_3d)
            points_2d_list.append(points_2d)
            weights_list.append(weights)
            rotations.append(rotation)
            centers.append(center)
            filenames.append(filename)

    # reduce error caused by incorrectness of depth map
    points, observations = refine_points(
            points_3d_list, points_2d_list, weights_list,
            rotations, centers, principal_point, focal_length)

    # save joint ball centers to json file
    X, Y, Z = points.T
    sel = ((X > xmin) & (np.abs(Y) > ymin) & (Z > zmin) &
           (X < xmax) & (np.abs(Y) < ymax) & (Z < zmax))
    points = points[sel]
    observations = observations[sel]
    structure = [ {'key': k,
                   'value': {
                       'X': point.tolist(),
                       'observations': [ {
                           'key': i,
                           'value': {
                               'id_feat': j,
                               'x': points_2d_list[i][j].tolist()}}
                       for i, j in observation ]}}
                  for k, (point, observation) in
                  enumerate(zip(points, observations)) ]
    outpath = os.path.dirname(os.path.abspath(outfile))
    if not os.path.exists(outpath): os.makedirs(outpath)
    with open(outfile, 'w') as fout:
        json.dump(OrderedDict([
                      ('sfm_data_version', '0.3'),
                      ('root_path', '../images'),
                      ('views', data['views']),
                      ('intrinsics', data['intrinsics']),
                      ('extrinsics', data['extrinsics']),
                      ('structure', structure),
                      ('control_points', [])]), fout, indent=4)

    # render joint balls to verify reconigntion
    outpath = os.path.splitext(os.path.abspath(outfile))[0]
    if not os.path.exists(outpath): os.makedirs(outpath)
    with ProgressBar(filenames, 'Rendering'.ljust(24)) as progress:
        ts = rmax * 2 + 3  # cell size
        tg = 1             # cell gap
        dx = tg + ts * w // h + 2
        dy = tg + 22
        summary = np.zeros((dy + (ts + tg) * len(data['views']),
                            dx + (ts + tg) * nrender, 3), dtype=np.uint8)

        # draw summary header
        for i, (x, y, z) in enumerate(points[:nrender]):
            tx = dx + (ts + tg) * i
            ty = tg
            cv2.putText(summary, str(i), (tx + 5, ty + 18),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                        color=(96, 96, 96), thickness=2, lineType=cv2.LINE_AA)
            tx += ts - 1
            draw_tiny_number(summary, '{:.4f}'.format(x), (tx, ty + 1),
                                      halign='right', valign='top')
            draw_tiny_number(summary, '{:.4f}'.format(y), (tx, ty + 8),
                                      halign='right', valign='top')
            draw_tiny_number(summary, '{:.4f}'.format(z), (tx, ty + 15),
                                      halign='right', valign='top')

        for j, (filename, rotation, center) in enumerate(zip(
                filenames, rotations, centers)):
            progress()
            image = cv2_imread(os.path.join(image_path, filename))

            # draw summary thumbnail
            th = ts
            tw = th * w // h
            tx = tg
            ty = dy + (ts + tg) * j
            summary[ty:ty+th,tx:tx+tw] = cv2.resize(
                    image, (tw, th), interpolation=cv2.INTER_AREA)
            draw_tiny_number(summary, os.path.splitext(filename)[0]
                                      .replace('mirror_', 'n_'), (tx+1, ty+1),
                                      halign='left', valign='top')

            # draw summary cells
            X, Y, Z = np.dot(rotation, np.subtract(points[:nrender], center).T)
            D = Z**2 - radius**2
            A = np.degrees(np.arctan2(Y, X))
            W = 2 * focal_length * radius * np.sqrt(X**2 + Y**2 + D) / D
            H = 2 * focal_length * radius / np.sqrt(D)
            X = X * Z * focal_length / D + principal_point[0]
            Y = Y * Z * focal_length / D + principal_point[1]
            for i, (x, y, ew, eh, a) in enumerate(zip(X, Y, W, H, A)):
                if j in set(map(itemgetter(0), observations[i])):
                    color = (0, 255, 255)
                else:
                    color = (255, 255, 255)
                sx = int(round(x - ts / 2))
                sy = int(round(y - ts / 2))
                tw = min(sx + ts, w) - max(sx, 0)
                th = min(sy + ts, h) - max(sy, 0)
                if tw <= 0 or th <= 0: continue
                tx = dx + (ts + tg) * i - min(sx, 0)
                ty = dy + (ts + tg) * j - min(sy, 0)
                sx = max(sx, 0)
                sy = max(sy, 0)
                summary[ty:ty+th,tx:tx+tw] = image[sy:sy+th,sx:sx+tw]
                cv2.ellipse(summary, ((x - sx + tx, y - sy + ty), (ew, eh), a),
                            color, 1, lineType=cv2.LINE_AA)

            # draw image
            for i, (x, y, ew, eh, a) in enumerate(zip(X, Y, W, H, A)):
                if j in set(map(itemgetter(0), observations[i])):
                    color = (0, 255, 255)
                else:
                    color = (255, 255, 255)
                cv2.ellipse(image, ((x, y), (ew, eh), a),
                            color, 1, lineType=cv2.LINE_AA)
                draw_tiny_number(image, str(i), (int(round(x)), int(round(y))))

            cv2_imwrite(os.path.join(outpath, filename), image)
        cv2_imwrite(os.path.join(outpath, 'summary.png'), summary)


def strut_main(bodyjson, bodyobj, gatejson, gateobj, outfile,
               body_joint_index=0, gate_joint_index=0,
               joint_diameter=0.01, strut_diameter=0.035,
               strut_offset=0.01, view_width=0.1,
               focal_length=2000, nsection=5):

    with ProgressBar(7, 'Loading'.ljust(24)) as progress:

        progress()
        with open(bodyjson, 'r') as fin:
            body_joint = np.array(json.load(fin)['structure']
                                  [body_joint_index]['value']['X'])
            body_joint[1] = np.abs(body_joint[1])

        progress()
        with open(gatejson, 'r') as fin:
            gate_joint = np.array(json.load(fin)['structure']
                                  [gate_joint_index]['value']['X'])
            gate_joint[1] = np.abs(gate_joint[1])

        body_joint_right = body_joint
        gate_joint_right = gate_joint
        body_joint_left = np.multiply(body_joint, [1, -1, 1])
        gate_joint_left = np.multiply(gate_joint, [1, -1, 1])

        progress()
        length = np.linalg.norm(body_joint - gate_joint)
        w = int(view_width * focal_length)
        h = int((length - strut_offset * 2) * focal_length / (nsection - 1))
        l = h * nsection
        renderer = MeshRenderer(w, l)

        progress()
        V, T, N, F, mtllib, usemtl = load_mesh_obj(bodyobj)

        progress()
        body_points, body_normals = get_face_normal(V, F)
        body_kdtree = cKDTree(body_points)
        renderer.add(V, T, N, F, mtllib, usemtl)

        progress()
        V, T, N, F, mtllib, usemtl = load_mesh_obj(gateobj)

        progress()
        gate_points, gate_normals = get_face_normal(V, F)
        gate_kdtree = cKDTree(gate_points)
        renderer.add(V, T, N, F, mtllib, usemtl)

    lstack = []
    mstack = []
    rstack = []
    jobs = [[(w, l), mstack, body_joint_left, gate_joint_left,
             [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], 0.5],
            [(w, l), mstack, body_joint_left, gate_joint_left,
             [[1, 0, 0], [0, 0, -1], [0, 1, 0]], 0.5],
            [(w, l), mstack, body_joint_right, gate_joint_right,
             [[1, 0, 0], [0, 0, -1], [0, 1, 0]], 0.5],
            [(w, l), mstack, body_joint_right, gate_joint_right,
             [[0, -1, 0], [0, 0, -1], [1, 0, 0]], 0.5]]
    for offset in np.linspace(0, 1, nsection):
        jobs.append([(w, h), lstack, body_joint_left, gate_joint_left,
                     [[1, 0, 0], [0, 1, 0], [0, 0, 1]], offset])
        jobs.append([(w, h), rstack, body_joint_right, gate_joint_right,
                     [[1, 0, 0], [0, 1, 0], [0, 0, 1]], offset])

    with ProgressBar(jobs, 'Rendering'.ljust(24)) as progress:

        for image_size, stack, body_joint, gate_joint, frame, offset in jobs:
            progress()

            principal_point = np.subtract(image_size, 1) / 2
            zaxis = body_joint - gate_joint
            zaxis /= np.linalg.norm(zaxis)
            yaxis = np.cross(zaxis, [0, 1, 0])
            yaxis /= np.linalg.norm(yaxis)
            xaxis = np.cross(yaxis, zaxis)
            center = np.add((body_joint - zaxis * strut_offset) * (1 - offset),
                            (gate_joint + zaxis * strut_offset) * offset)
            rotation = np.dot(frame, [xaxis, yaxis, zaxis])

            Y, X = np.mgrid[:image_size[1],:image_size[0]]
            X = X.ravel()
            Y = Y.ravel()
            P = np.transpose([X - principal_point[0],
                              Y - principal_point[1],
                              np.zeros(X.shape)]) / focal_length
            P = np.dot(P, rotation) + center
            Ib = body_kdtree.query(P)[1]
            Ig = gate_kdtree.query(P)[1]
            Sb = (np.sum(np.multiply(body_points[Ib] - P,
                                     body_normals[Ib]), axis=1) > 0)
            Sg = (np.sum(np.multiply(gate_points[Ig] - P,
                                     gate_normals[Ig]), axis=1) > 0)

            image = renderer.render(image_size, focal_length, rotation, center,
                                    znear=0, zfar=0.1, ortho=True)
            depth = renderer.render(image_size, focal_length, rotation, center,
                                    znear=0, zfar=0.1, ortho=True, depth=True)
            image = np.clip(image * np.clip(1 - depth[:,:,None] / 0.1, 0, 1),
                            0, 255).astype(np.uint8)

            Ss = np.zeros(image_size[::-1], dtype=np.uint8)
            if stack is mstack:
                p1 = np.dot(rotation, body_joint - center)[:2]
                p2 = np.dot(rotation, gate_joint - center)[:2]
                p3 = p1 + [strut_diameter / 2, strut_offset]
                p4 = p2 - [strut_diameter / 2, strut_offset]
                p1 = tuple(map(int, map(round,
                         p1 * focal_length + principal_point)))
                p2 = tuple(map(int, map(round,
                         p2 * focal_length + principal_point)))
                p3 = tuple(map(int, map(round,
                         p3 * focal_length + principal_point)))
                p4 = tuple(map(int, map(round,
                         p4 * focal_length + principal_point)))
                cv2.circle(Ss, p1, 10, 1, -1)
                cv2.circle(Ss, p2, 10, 1, -1)
                cv2.rectangle(Ss, p3, p4, 1, -1)
            else:
                p1 = tuple(map(int, map(round, principal_point)))
                r = int(strut_diameter * focal_length / 2)
                cv2.circle(Ss, p1, r, 1, -1)
            Ss = Ss.ravel().astype(bool)

            image[Y[Sb|Sg|Ss],X[Sb|Sg|Ss]] >>= 1
            image[Y[Sb],X[Sb],0] |= 0x80
            image[Y[Ss],X[Ss],1] |= 0x80
            image[Y[Sg],X[Sg],2] |= 0x80

            edges = np.maximum(np.maximum(
                        cv2.Canny(Sb.reshape(image_size[::-1])
                                    .astype(np.uint8) * 255, 100, 200),
                        cv2.Canny(Ss.reshape(image_size[::-1])
                                    .astype(np.uint8) * 255, 100, 200)),
                        cv2.Canny(Sg.reshape(image_size[::-1])
                                    .astype(np.uint8) * 255, 100, 200))
            image = np.maximum(image, edges[:,:,None])

            image[:1,:] = 0
            image[-1:,:] = 0
            image[:,:1] = 0
            image[:,-1:] = 0
            stack.append(image)

    mstack.insert(0, np.concatenate(lstack))
    mstack.append(np.concatenate(rstack))
    image = np.hstack(mstack)
    image[h//2::h,::4,1] |= 0x80
    cv2_imwrite(outfile, image)


def render_main(jsonfile, infile=None, outpath=None, znear=0.001):
    if outpath is None:
        outpath = os.path.splitext( infile if infile else jsonfile )[0]
    if not os.path.exists(outpath): os.makedirs(outpath)

    with ProgressBar(3, 'Loading'.ljust(24)) as progress:

        progress()
        with open(jsonfile) as fin:
            data = json.load(fin)
        intrinsic_data = data['intrinsics'][0]['value']['ptr_wrapper']['data']
        focal_length = intrinsic_data['focal_length']
        principal_point = intrinsic_data['principal_point']
        image_size = intrinsic_data['width'], intrinsic_data['height']
        image_path = os.path.join(os.path.dirname(jsonfile), data['root_path'])

        progress()
        ext = os.path.splitext(infile)[1].lower() if infile else None

        if not ext:
            points = []
            point_keys = []
            observations = defaultdict(set)
            for i, point in enumerate(data['structure']):
                point_keys.append(point['key'])
                points.append(point['value']['X'])
                for observation in point['value']['observations']:
                    observations[observation['key']].add(i)
            if not points: raise ValueError

            progress()
            points = np.array(points)

        elif ext == '.ply':
            points = load_points_ply(infile)

            progress()
            render = lambda rotation, center: render_points(
                    points, image_size, focal_length, rotation, center,
                    znear=znear)

        elif ext == '.obj':
            V, T, N, F, mtllib, usemtl = load_mesh_obj(infile)

            progress()
            renderer = MeshRenderer(*image_size)
            renderer.add(V, T, N, F, mtllib, usemtl)
            render = lambda rotation, center: renderer.render(
                    image_size, focal_length, rotation, center)

        else:
            raise NotImplementedError

    with ProgressBar(data['views'], 'Rendering'.ljust(24)) as progress:
        for view, extrinsic in zip(data['views'], data['extrinsics']):
            progress()

            filename = view['value']['ptr_wrapper']['data']['filename']
            raw = cv2_imread(os.path.join(image_path, filename))

            if not infile:
                X, Y, Z = np.dot(extrinsic['value']['rotation'],
                                 (points - extrinsic['value']['center']).T)
                for i in np.argsort(Z)[::-1]:
                    z = Z[i]
                    if z < znear: break
                    draw_tiny_number(
                            raw, str(point_keys[i]),
                            (X[i] / z * focal_length + principal_point[0],
                             Y[i] / z * focal_length + principal_point[1]),
                            color=( (0, 255, 255)
                                    if i in observations[view['key']] else
                                    None ))
                cv2_imwrite(os.path.join(outpath, filename), raw)
                continue

            image = render(extrinsic['value']['rotation'],
                           extrinsic['value']['center'])
            cv2_imwrite(os.path.join(outpath, filename), image)

            filename = os.path.splitext(filename)[0] + '_mixed.jpg'
            cv2_imwrite(os.path.join(outpath, filename),
                        ((image.astype(int) +
                          raw.astype(int)) >> 1).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version',
                        action='version', version=__version__)
    subparsers = parser.add_subparsers(dest='command')

    subparser = subparsers.add_parser('calibrate')
    subparser.add_argument('-c', '--cross-validation', action='store_true',
                           help='run cross validation after calibration')
    subparser.add_argument('camerafile',
                           help='path of output camera.json')
    subparser.add_argument('paths', nargs='+',
                           help='path of images to calibrate camera')

    subparser = subparsers.add_parser('import')
    subparser.add_argument('-s', '--stage', metavar='N', type=int, default=10,
                           help='num of iters to run adjust (default = %(default)s)')
    subparser.add_argument('-e', '--expand', metavar='N', type=int, default=0,
                           help='num of iters to run expand (default = %(default)s)')
    subparser.add_argument('-m', '--mirror', action='store_true',
                           help='apply mirror operation in run.bat')
    subparser.add_argument('camerafile',
                           help='path of input camera.json')
    subparser.add_argument('paths', nargs='+',
                           help='path of images to reconstruct liftgate')

    subparser = subparsers.add_parser('adjust')
    subparser.add_argument('infile',
                           help='path of input sfm_data.json')
    subparser.add_argument('outfile', nargs='?',
                           help='path of output sfm_data.json')

    subparser = subparsers.add_parser('expand')
    subparser.add_argument('-p', '--pcc-threshold', metavar='PCC',
                           type=float, default=0.4,
                           help='PCC threshold to accept (default = %(default)s)')
    subparser.add_argument('-o', '--optimize', action='store_true',
                           help='refine camera position')
    subparser.add_argument('infile',
                           help='path of input sfm_data.json')
    subparser.add_argument('plyfile',
                           help='path of reference ply file')
    subparser.add_argument('outfile', nargs='?',
                           help='path of output sfm_data.json')

    subparser = subparsers.add_parser('align')
    subparser.add_argument('-c', '--clip', action='store_true',
                           help='remove insignificant points')
    subparser.add_argument('paths', nargs='+',
                           help='path of sfm_data.json and ply files')

    subparser = subparsers.add_parser('mirror')
    subparser.add_argument('infile',
                           help='path of input sfm_data.json')
    subparser.add_argument('outfile',
                           help='path of output sfm_data.json')

    subparser = subparsers.add_parser('clip')
    subparser.add_argument('-xmin', metavar='XMIN', type=float, default=-2,
                           help='default = %(default)s')
    subparser.add_argument('-xmax', metavar='XMAX', type=float, default=2,
                           help='default = %(default)s')
    subparser.add_argument('-ymin', metavar='YMIN', type=float, default=-2,
                           help='default = %(default)s')
    subparser.add_argument('-ymax', metavar='YMAX', type=float, default=2,
                           help='default = %(default)s')
    subparser.add_argument('-zmin', metavar='ZMIN', type=float, default=-2,
                           help='default = %(default)s')
    subparser.add_argument('-zmax', metavar='ZMAX', type=float, default=2,
                           help='default = %(default)s')
    subparser.add_argument('infile',
                           help='path of input ply/obj file')
    subparser.add_argument('outfile',
                           help='path of output ply/obj file')

    subparser = subparsers.add_parser('close')
    subparser.add_argument('bodyobj',
                           help='path of input body obj file')
    subparser.add_argument('gateobj',
                           help='path of input gate obj file')
    subparser.add_argument('gatejson', nargs='?',
                           help='path of input gate sfm_data.json')

    subparser = subparsers.add_parser('joint')
    subparser.add_argument('-xmin', metavar='XMIN', type=float, default=0,
                           help='default = %(default)s')
    subparser.add_argument('-xmax', metavar='XMAX', type=float, default=0.6,
                           help='default = %(default)s')
    subparser.add_argument('-ymin', metavar='YMIN', type=float, default=0.45,
                           help='default = %(default)s')
    subparser.add_argument('-ymax', metavar='YMAX', type=float, default=0.7,
                           help='default = %(default)s')
    subparser.add_argument('-zmin', metavar='ZMIN', type=float, default=-0.7,
                           help='default = %(default)s')
    subparser.add_argument('-zmax', metavar='ZMAX', type=float, default=-0.05,
                           help='default = %(default)s')
    subparser.add_argument('infile',
                           help='path of input sfm_data.json')
    subparser.add_argument('objfile',
                           help='path of reference obj file')
    subparser.add_argument('outfile', nargs='?',
                           help='path of output sfm_data.json')

    subparser = subparsers.add_parser('strut')
    subparser.add_argument('-b', '--body-joint-index',
                           metavar='N', type=int, default=0,
                           help='body joint point index in bodyjson')
    subparser.add_argument('-g', '--gate-joint-index',
                           metavar='N', type=int, default=0,
                           help='gate joint point index in gatejson')
    subparser.add_argument('bodyjson',
                           help='path of input body sfm_data.json')
    subparser.add_argument('bodyobj',
                           help='path of input body obj file')
    subparser.add_argument('gatejson',
                           help='path of input gate sfm_data.json')
    subparser.add_argument('gateobj',
                           help='path of input gate obj file')
    subparser.add_argument('outfile',
                           help='path of output image')

    subparser = subparsers.add_parser('render')
    subparser.add_argument('jsonfile',
                           help='path of input sfm_data.json')
    subparser.add_argument('infile', nargs='?',
                           help='path of input ply/obj file')
    subparser.add_argument('outpath', nargs='?',
                           help='path to save render image')

    args = parser.parse_args()

    if args.command == 'calibrate':
        if not args.camerafile.lower().endswith('.json'): raise ValueError
        calibrate_main(args.camerafile, args.paths, args.cross_validation)
    elif args.command == 'import':
        if not args.camerafile.lower().endswith('.json'): raise ValueError
        import_main(args.camerafile, args.paths,
                    stage=args.stage, expand=args.expand, mirror=args.mirror)
    elif args.command == 'adjust':
        if not args.infile.lower().endswith('.json'): raise ValueError
        outfile = args.outfile if args.outfile is not None else args.infile
        if not outfile.lower().endswith('.json'): raise ValueError
        adjust_main(args.infile, outfile)
    elif args.command == 'expand':
        if not args.infile.lower().endswith('.json'): raise ValueError
        if not args.plyfile.lower().endswith('.ply'): raise ValueError
        outfile = args.outfile if args.outfile is not None else args.infile
        if not outfile.lower().endswith('.json'): raise ValueError
        expand_main(args.infile, args.plyfile, outfile,
                    args.pcc_threshold, args.optimize)
    elif args.command == 'align':
        if len(args.paths) % 2 != 0: raise ValueError
        jsonfiles, plyfiles = zip(*zip(args.paths[0::2], args.paths[1::2]))
        if not all( jsonfile.lower().endswith('.json')
                    for jsonfile in jsonfiles ): raise ValueError
        if not all( plyfile.lower().endswith('.ply')
                    for plyfile in plyfiles ): raise ValueError
        align_main(jsonfiles, plyfiles, args.clip)
    elif args.command == 'mirror':
        if not args.infile.lower().endswith('.json'): raise ValueError
        if not args.outfile.lower().endswith('.json'): raise ValueError
        mirror_main(args.infile, args.outfile)
    elif args.command == 'clip':
        inext = os.path.splitext(args.infile)[1].lower()
        outext = os.path.splitext(args.outfile)[1].lower()
        if inext not in ('.ply', '.obj'): raise ValueError
        if outext not in ('.ply', '.obj'): raise ValueError
        if inext != outext: raise ValueError
        clip_main(args.infile, args.outfile,
                  xmin=args.xmin, ymin=args.ymin, zmin=args.zmin,
                  xmax=args.xmax, ymax=args.ymax, zmax=args.zmax)
    elif args.command == 'close':
        if not args.bodyobj.lower().endswith('.obj'): raise ValueError
        if not args.gateobj.lower().endswith('.obj'): raise ValueError
        if args.gatejson:
            if not args.gatejson.lower().endswith('.json'): raise ValueError
        close_main(args.bodyobj, args.gateobj, args.gatejson)
    elif args.command == 'joint':
        if not args.infile.lower().endswith('.json'): raise ValueError
        if not args.objfile.lower().endswith('.obj'): raise ValueError
        outfile = args.outfile if args.outfile is not None else \
                  os.path.splitext(args.infile)[0] + '_joint.json'
        if not outfile.lower().endswith('.json'): raise ValueError
        joint_main(args.infile, args.objfile, outfile,
                   xmin=args.xmin, ymin=args.ymin, zmin=args.zmin,
                   xmax=args.xmax, ymax=args.ymax, zmax=args.zmax)
    elif args.command == 'strut':
        if not args.bodyjson.lower().endswith('.json'): raise ValueError
        if not args.bodyobj.lower().endswith('.obj'): raise ValueError
        if not args.gatejson.lower().endswith('.json'): raise ValueError
        if not args.gateobj.lower().endswith('.obj'): raise ValueError
        strut_main(args.bodyjson, args.bodyobj, args.gatejson, args.gateobj,
                   args.outfile, args.body_joint_index, args.gate_joint_index)
    elif args.command == 'render':
        if not args.jsonfile.lower().endswith('.json'): raise ValueError
        if args.infile:
            inext = os.path.splitext(args.infile)[1].lower()
            if inext not in ('.ply', '.obj'): raise ValueError
        render_main(args.jsonfile, args.infile, args.outpath)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
