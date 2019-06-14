#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: eph

from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.optimize import minimize
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    from scipy.ndimage.filters import gaussian_filter


__version__ = '1.5.180312'


def load(angle, path, configs=None):
    if configs is None: configs = []

    for fn in os.listdir(path):
        if not fn.endswith('.csv'): continue
        fn = os.path.join(path, fn)

        for ext in ('png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP'):
            if not os.path.isfile(fn[:-3] + ext): continue
            image = plt.imread(fn[:-3] + ext)
            if len(image.shape) == 2:  # gray -> RGB
                image = np.repeat(np.expand_dims(image, 2), 3, 2)
            if image.shape[2] > 3:  # RGBA -> RGB
                image = image[:,:,:3]
            if image.dtype == np.uint8:  # uint8 -> float
                image = image.astype(float) / 255
            break
        else: continue

        h, w = image.shape[:2]
        config = {'file': fn, 'image': image}
        configs.append(config)
        cx = (w - 1) / 2
        cy = (h - 1) / 2

        points = []
        mask = []
        with open(fn) as fin:
            for i, line in enumerate(fin):
                line = line.strip()
                if line:
                    x, y = map(float, line.split(','))
                    points.append([x - cx, y - cy])
                    mask.append(i)
        config['points'] = np.array(points).T
        config['mask'] = mask
        config['angle'] = angle

    return configs


def rotate_points(points, gate_angle):
    sg = np.sin(gate_angle); cg = np.cos(gate_angle)
    X, Y, Z, I = points[:,10:]
    return np.hstack([points[:,:10], [X, cg * Y + sg * Z, cg * Z - sg * Y, I]])


def fit(configs, board_width, board_height, asymmetry=False):
    num_points = max( max(config['mask']) for config in configs ) + 1

    rw = board_width / 2
    rh = board_height / 2

    def get_camera(iK, scale):
        x = next(iK); y = next(iK); z = next(iK)
        a = next(iK); b = next(iK); c = next(iK)
        aa = a * a; bb = b * b; cc = c * c
        d = np.sqrt(aa + bb + cc) / np.pi
        g = np.sinc(d)
        h = np.sinc(d / 2)**2 / 2
        ag = a * g; abh = a * b * h
        bg = b * g; ach = a * c * h
        cg = c * g; bch = b * c * h
        return np.dot(np.diag([scale, scale, 1., 1.]),
                      [[1. - (bb + cc) * h, abh - cg, ach + bg, x],
                       [abh + cg, 1. - (aa + cc) * h, bch - ag, y],
                       [ach - bg, bch + ag, 1. - (aa + bb) * h, z],
                       [0., 0., 0., 1.]])

    def get_points(iK):
        rx = next(iK); ry = next(iK); rz = next(iK)
        ra = next(iK); sa = np.sin(ra); ca = np.cos(ra)
        rb = next(iK); sb = np.sin(rb); cb = np.cos(rb)
        px = next(iK); qx = next(iK)
        lbx = next(iK); lby = next(iK); lbz = next(iK)
        lgx = next(iK); lgy = next(iK); lgz = next(iK)
        if asymmetry:
            rbx = next(iK); rby = next(iK); rbz = next(iK)
            rgx = next(iK); rgy = next(iK); rgz = next(iK)
        else:
            rbx = -lbx; rby = lby; rbz = lbz
            rgx = -lgx; rgy = lgy; rgz = lgz
        rwca = rw * ca; rwsa = rw * sa; rhca = rh * ca; rhsa = rh * sa
        rw1 = rwca - rhsa; rh1 = rhca + rwsa
        rw2 = rwca + rhsa; rh2 = rhca - rwsa
        return np.array([[rx - cb * rw1, ry - rh1, rz + sb * rw1, 1.],
                         [rx + cb * rw2, ry - rh2, rz - sb * rw2, 1.],
                         [rx - cb * rw2, ry + rh2, rz + sb * rw2, 1.],
                         [rx + cb * rw1, ry + rh1, rz - sb * rw1, 1.],
                         [-px, 0., 0., 1.],
                         [ px, 0., 0., 1.],
                         [-qx, 0., 0., 1.],
                         [ qx, 0., 0., 1.],
                         [lbx, lby, lbz, 1.],
                         [rbx, rby, rbz, 1.],
                         [lgx, lgy, lgz, 1.],
                         [rgx, rgy, rgz, 1.]] +
                        [ [next(iK), next(iK), next(iK), 1.]
                          for i in range(num_points - 12) ]).T

    def printres(K, score):
        iK = iter(K)
        print('Board:  {:4.0f} mm, {:4.0f} mm, {:4.0f} mm,'
                     ' {:4.2f} deg, {:4.2f} deg'.format(
                      1000 * next(iK), 1000 * next(iK), 1000 * next(iK),
                      np.degrees(next(iK)), np.degrees(next(iK))))
        print('Hinge:  {:7.2f} mm, {:7.2f} mm'.format(
                      1000 * next(iK), 1000 * next(iK)))
        lbx = 1000 * next(iK); lby = 1000 * next(iK); lbz = 1000 * next(iK)
        lgx = 1000 * next(iK); lgy = 1000 * next(iK); lgz = 1000 * next(iK)
        if asymmetry:
            rbx = 1000 * next(iK); rby = 1000 * next(iK); rbz = 1000 * next(iK)
            rgx = 1000 * next(iK); rgy = 1000 * next(iK); rgz = 1000 * next(iK)
        else:
            rbx = -lbx; rby = lby; rbz = lbz
            rgx = -lgx; rgy = lgy; rgz = lgz
        print('Body ball joint:  {:7.2f} mm, {:7.2f} mm, {:7.2f} mm  (left)'
              .format(lbx, lby, lbz))

        print('                  {:7.2f} mm, {:7.2f} mm, {:7.2f} mm  (right)'
              .format(rbx, rby, rbz))
        print('Gate ball joint:  {:7.2f} mm, {:7.2f} mm, {:7.2f} mm  (left, closed)'
              .format(lgx, lgy, lgz))
        print('                  {:7.2f} mm, {:7.2f} mm, {:7.2f} mm  (right, closed)'
              .format(rgx, rgy, rgz))
        for i in range(num_points - 12):
            print('Point {:d}:  {:7.2f} mm, {:7.2f} mm, {:7.2f} mm  (closed)'
                  .format(13 + i, 1000 * next(iK), 1000 * next(iK), 1000 * next(iK)))
        print('相机视频demo scale:  {:.3f} pixel/mm'.format(np.exp(next(iK)) / 1000))
        for config in configs:
            print('相机视频demo:  {:4.0f} mm, {:4.0f} mm, {:4.0f} mm,'
                          ' {:4.0f} deg, {:4.0f} deg, {:4.0f} deg'.format(
                      1000 * next(iK), 1000 * next(iK),
                      1000 * next(iK), np.degrees(next(iK)),
                      np.degrees(next(iK)), np.degrees(next(iK))))
        print('RMS error:  {:g} pixel'.format(score))
        print('Ball joint distance: {:7.2f} mm (left body ~ right body)'
              .format(np.sqrt((lbx - rbx)**2 + (lby - rby)**2 + (lbz - rbz)**2)))
        print('                     {:7.2f} mm (left gate ~ right gate)'
              .format(np.sqrt((lgx - rgx)**2 + (lgy - rgy)**2 + (lgz - rgz)**2)))
        print('                     {:7.2f} mm (left body ~ left gate, closed)'
              .format(np.sqrt((lbx - lgx)**2 + (lby - lgy)**2 + (lbz - lgz)**2)))
        print('                     {:7.2f} mm (right body ~ right gate, closed)'
              .format(np.sqrt((rbx - rgx)**2 + (rby - rgy)**2 + (rbz - rgz)**2)))
        print('                     {:7.2f} mm (left body ~ right gate, closed)'
              .format(np.sqrt((lbx - rgx)**2 + (lby - rgy)**2 + (lbz - rgz)**2)))
        print('                     {:7.2f} mm (right body ~ left gate, closed)'
              .format(np.sqrt((rbx - lgx)**2 + (rby - lgy)**2 + (rbz - lgz)**2)))
        for gate_angle in sorted(set( config['angle'] for config in configs )):
            sg = np.sin(gate_angle); cg = np.cos(gate_angle)
            lgy2 = sg * lgz + cg * lgy; lgz2 = cg * lgz - sg * lgy
            rgy2 = sg * rgz + cg * rgy; rgz2 = cg * rgz - sg * rgy
            print('                     {:7.2f} mm (left body ~ left gate, {:.2f} deg)'
                  .format(np.sqrt((lbx - lgx)**2 + (lby - lgy2)**2 + (lbz - lgz2)**2),
                          np.degrees(gate_angle)))
            print('                     {:7.2f} mm (right body ~ right gate, {:.2f} deg)'
                  .format(np.sqrt((rbx - rgx)**2 + (rby - rgy2)**2 + (rbz - rgz2)**2),
                          np.degrees(gate_angle)))
            print('                     {:7.2f} mm (left body ~ right gate, {:.2f} deg)'
                  .format(np.sqrt((lbx - rgx)**2 + (lby - rgy2)**2 + (lbz - rgz2)**2),
                          np.degrees(gate_angle)))
            print('                     {:7.2f} mm (right body ~ left gate, {:.2f} deg)'
                  .format(np.sqrt((rbx - lgx)**2 + (rby - lgy2)**2 + (rbz - lgz2)**2),
                          np.degrees(gate_angle)))

    def cost(K):
        score = 0.
        count = 0
        iK = iter(K)
        points = get_points(iK)
        scale = np.exp(next(iK))
        for config in configs:
            U, V = config['points']
            mask = config['mask']
            X, Y, Z, _ = np.dot(get_camera(iK, scale),
                                rotate_points(points, config['angle'])[:,mask])
            X = np.divide(X, Z)
            Y = np.divide(Y, Z)
            score += np.sum(np.square(X - U))
            score += np.sum(np.square(Y - V))
            count += len(mask)
        return np.sqrt(score / count)

    K0 = np.array([0., 0.5, 0., 0., 0.,
                   0.3, 0.5,
                   -0.5, 0.1, 0.,
                   -0.6, 0.4, -0.3] +
                  [0., 0.8, -0.5] * (num_points - 12) +
                  [8.] +
                  [0., 0., 2.,
                   -0.2, 0., 0.] * len(configs))

    if asymmetry:

        # first stage
        asymmetry = False
        res = minimize(cost, K0,
                       method='SLSQP', options=dict(maxiter=sys.maxint))
        assert res.success

        # second stage
        asymmetry = True
        K0 = np.concatenate([res.x[:10], res.x[7:13], res.x[10:]])

    res = minimize(cost, K0,
                   method='SLSQP', options=dict(maxiter=sys.maxint))
    assert res.success
    printres(res.x, res.fun)

    iK = iter(res.x)
    points = get_points(iK)
    scale = np.exp(next(iK))
    cameras = [ get_camera(iK, scale) for config in configs ]
    return cameras, points


def save(configs, cameras, points):
    for config, camera in zip(configs, cameras):
        image = config['image']

        dpi = 100
        h, w = image.shape[:2]
        fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                            wspace=0, hspace=0)
        ax = fig.add_subplot(1, 1, 1)
        ax.patch.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-w / 2, w / 2)
        ax.set_ylim(h / 2, -h / 2)

        P = rotate_points(points, config['angle'])
        X, Y, Z, _ = np.dot(camera, P)
        X = np.divide(X, Z)
        Y = np.divide(Y, Z)
        A = [6, 8, 10, 6, 4, 5, 7, 9, 11, 7]  # car indices
        B = [0, 1, 3, 2, 0]  # board indices
        ax.plot(X[B], Y[B], 'k', lw=1, zorder=0)
        ax.plot(X[A], Y[A], 'k', lw=1, zorder=0)
        ax.plot(X[B], Y[B], 'o', ms=35, mew=1, mfc='w', mec='k', zorder=1)
        ax.plot(X[A], Y[A], 'o', ms=35, mew=1, mfc='w', mec='k', zorder=1)
        ax.plot(X[12:], Y[12:], 'o', ms=35, mew=1, mfc='w', mec='k', zorder=1)

        D = {'n':  {'pre': '', 'suf': '\n', 'ha': 'center', 'va': 'bottom'},
             'en': {'pre': '  ', 'suf': '\n', 'ha': 'left', 'va': 'bottom'},
             'e':  {'pre': '     ', 'suf': '', 'ha': 'left', 'va': 'center'},
             'es': {'pre': '\n  ', 'suf': '', 'ha': 'left', 'va': 'top'},
             's':  {'pre': '\n', 'suf': '', 'ha': 'center', 'va': 'top'},
             'ws': {'pre': '\n', 'suf': '  ', 'ha': 'right', 'va': 'top'},
             'w':  {'pre': '', 'suf': '     ', 'ha': 'right', 'va': 'center'},
             'wn': {'pre': '', 'suf': '  \n', 'ha': 'right', 'va': 'bottom'}}
        for i, (u, v, (x, y, z, _), loc1, loc2) in enumerate(zip(X, Y,
                P.T, 'ws es wn en e w wn en ws es w e'.split(),
                     'es ws en wn en wn w e es ws e w'.split())):
            ax.text(u, v, '{!s}{:.0f}, {:.0f}, {:.0f}{!s}'.format(
                                  D[loc2]['pre'], 1000 * x, 1000 * y,
                                  1000 * z, D[loc2]['suf']), color='k',
                    fontsize=24, ha=D[loc2]['ha'], va=D[loc2]['va'], zorder=2)
            ax.text(u, v, D[loc1]['pre'] + str(i + 1) + D[loc1]['suf'],
                    color='w', fontsize=24, fontweight='bold',
                    ha=D[loc1]['ha'], va=D[loc1]['va'], zorder=3)
            for u, v in [(u - 2, v - 2), (u, v - 2), (u + 2, v - 2),
                         (u - 2, v),                 (u + 2, v),
                         (u - 2, v + 2), (u, v + 2), (u + 2, v + 2)]:
                ax.text(u, v, D[loc1]['pre'] + str(i + 1) + D[loc1]['suf'],
                        color='k', fontsize=24, fontweight='bold',
                        ha=D[loc1]['ha'], va=D[loc1]['va'], zorder=2)

        X, Y = config['points']
        ax.plot(X, Y, 'ok', ms=30, mfc='w', mec='k', zorder=2)
        ax.plot(X, Y, '+k', ms=30, mew=1, mec='k', zorder=3)
        ax.plot(X, Y, 'ow', ms=10, mfc='w', mec='None', zorder=4)

        X, Y, Z, _ = np.dot(camera, np.array([[0.0, 0.0, 0.0, 1.0],
                                              [0.1, 0.0, 0.0, 1.0],
                                              [0.0, 0.1, 0.0, 1.0],
                                              [0.0, 0.0, 0.1, 1.0]]).T)
        X = np.divide(X, Z)
        Y = np.divide(Y, Z)
        ax.plot(X[[0,1]], Y[[0,1]], 'k', lw=4, zorder=5)
        ax.plot(X[[0,2]], Y[[0,2]], 'k', lw=4, zorder=5)
        ax.plot(X[[0,3]], Y[[0,3]], 'k', lw=4, zorder=5)
        ax.plot(X[[1,2,3]], Y[[1,2,3]], 'o',
                ms=30, mfc='k', mec='None', zorder=5)

        ax.text(X[1], Y[1], 'x', fontsize=24, fontweight='bold',
                color='w', ha='center', va='center', zorder=6)
        ax.text(X[2], Y[2], 'y', fontsize=24, fontweight='bold',
                color='w', ha='center', va='center', zorder=6)
        ax.text(X[3], Y[3], 'z', fontsize=24, fontweight='bold',
                color='w', ha='center', va='center', zorder=6)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        mask = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8,
                             sep='').reshape((h, w, 3)) / 255
        fig.clear()
        fig.figimage(mask * image + (1 - mask) *
            gaussian_filter((image < 0.5).astype(float), 1))

        fig.savefig(config['file'][:-4] + '_fit.png')
        plt.close(fig)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version',
                        version=__version__)
    parser.add_argument('-a', '--asymmetry', action='store_true',
                        help='ball joint asymmetry considered')
    parser.add_argument('board_width', metavar='width', type=float,
                        help='board width in mm')
    parser.add_argument('board_height', metavar='height', type=float,
                        help='board height in mm')
    parser.add_argument('gate_angle', metavar='angle', type=float,
                        help='gate open angle in deg')
    parser.add_argument('path', metavar='path', nargs='?', default='.',
                        help='working directory')
    parser.add_argument('rest', metavar='...', nargs=argparse.REMAINDER,
                        help='additional angle-path pairs')
    args = parser.parse_args()

    angles = [args.gate_angle]
    paths = [args.path]
    for i, item in enumerate(args.rest):
        if i % 2 == 0:
            angles.append(float(item))
        else:
            paths.append(item)
    if len(angles) > len(paths): paths.append('.')

    configs = []
    for angle, path in zip(angles, paths):
        configs = load(np.radians(angle), path, configs)
    cameras, points = fit(configs,
                          args.board_width / 1000,
                          args.board_height / 1000,
                          args.asymmetry)
    save(configs, cameras, points)
