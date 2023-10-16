import collections
import copy
import datetime
import time
import gc

import numpy as np


IrcTuple = collections.namedtuple('IrcTupe', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', 'X, Y, Z')


def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)


def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)

    cri_a = ((coord_a - origin_a)) @ np.linalg.inv(direction_a) / vxSize_a
    cri_a = np.round(cri_a)

    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))
