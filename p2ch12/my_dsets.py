import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.funtional as F
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch12_raw')

CandidateInforTuple = namedtuple(
    'CandidateInfoTuple', 'isNodule_bool, diameter_mm, series_uid, center_xyz')


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open('data/part2/luna/annotations.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm))

    candidateInfo_list = []
    with open('data/part2/luna/candidates.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(
                        candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break
            candidateInfo_list.append(CandidateInforTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


class Ct:
    def __init_(self, series_uid):
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz,
                             self.vxSize_xyz, self.direction_a)
        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = Ct.getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    return ct_chunk, center_irc


def getCtAugumentCandidate(augmentation_dict, series_uid, center_xyz, witth_irc, use_cache=True):
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(
            series_uid, center_xyz, witth_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, witth_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1
        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = random.random() * 2 - 1
            transform_t[i, 3] = offset_float * random_float
        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = random.random() * 2 - 1
            transform_t[i, i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:  # RC平面のみ回転を適用する
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    transform_t = rotation_t @ transform_t

    affine_t = F.affine_grid(  # N x 3 x 4 for 3D
        transform_t[:3].unsqueeze(0).to(torch.float32),
        ct_t.size(),
        align_corners=False
    )

    augmented_chunk = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode='border',
        align_corners=False
    ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    # augmented_chunk[0]はshapeが(1, 1, 48, 48, 48)なので、shapeが(48, 48, 48)になるように[0]をつける
    return augmented_chunk[0], center_irc
