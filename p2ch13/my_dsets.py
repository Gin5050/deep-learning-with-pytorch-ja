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
import scipy.ndimage.morphology as morph

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch13_raw')

MaskTuple = namedtuple(
    'MaskTuple',
    'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask'
)

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz'
)


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []
    with open('data/part2/luna/annotation_with_malignancy.csv', 'r') as f:  # 結節=1はすべてこの中に入ってる
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            isMal_bool = {'False': False, 'True': True}[row[5]]

            candidateInfo_list.append(
                CandidateInfoTuple(
                    True,  # 結節
                    True,  # アノテーション
                    isMal_bool,  # 悪性
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz
                )
            )

    with open('data/part2/luna/candidate.csv') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool:
                candidateInfo_list.append(CandidateInfoTuple(
                    False,
                    False,
                    False,
                    0.0,
                    series_uid,
                    candidateCenter_xyz
                ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(
            candidateInfo_tup.series_uid, []).append(candidateInfo_tup)

    return candidateInfo_dict


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)  # mhd型に一旦して
        self.hu_a = np.array(sitk.GetArrayFromImage(
            ct_mhd), dtype=np.float32)  # numpy.arrayに落とし込む

        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(*ct_mhd.GetDirection()).reshape(3, 3)

        candidateInfo_list = getCandidateInfoDict()[self.series_uid]

        self.positiveInfo_list = [  # 実際に結節である場合のみの結節周囲データ
            candidate_tup
            for candidate_tup in candidateInfo_list
            if candidate_tup.isNodule_bool
        ]

        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        self.positive_indexes = (
            self.positive_mask.sum(axis=(1, 2)).nonzero()[0].tolist()
        )

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu=-700):
        boundingBox_a = np.zeros_like(
            self.hu_a, dtype=np.bool)  # 一つのCT全体のマスクを0で初期化

        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while (
                    self.hu_a[ci + index_radius, cr, cc] > threshold_hu
                    and self.hu_a[ci - index_radius, cr, cc] > threshold_hu
                ):
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while (
                    self.hu_a[ci, cr + row_radius, cc] > threshold_hu
                    and self.hu_a[ci, cr - row_radius, cc] > threshold_hu
                ):
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while (
                    self.hu_a[ci, cr, cc + col_radius] > threshold_hu
                    and self.hu_a[ci, cr, cc - col_radius] > threshold_hu
                ):
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # assert index_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            # assert row_radius > 0
            # assert col_radius > 0

            boundingBox_a[
                ci - index_radius: ci + index_radius + 1,
                cr - row_radius: cr + row_radius + 1,
                cc - col_radius: cc + col_radius + 1,
            ] = True

        mask_a = boundingBox_a & (self.hu_a > threshold_hu)  # 矩形領域の角を取ることができる

        return mask_a

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [
                    self.series_uid,
                    center_xyz,
                    self.origin_xyz,
                    self.vxSize_xyz,
                    center_irc,
                    axis,
                ]
            )

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    # series_uidで指定されたある人のCTデータ全体に対して結節マスクを含む情報を取得、
    # CTの値をクリップしてから結節周囲のデータ、結節マスクおよび結節の中心座標を返す
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc


@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


class Luna2dSegmentationDataset(Dataset):
    def __init__(
            self,
            val_stride=0,
            isValSet_bool=None,
            series_uid=None,
            contextSlices_count=3,
            fullCt_bool=False
    ):
        self.contextSlices_count = contextSlices_count  # 上下何枚チャネル取るか
        self.fullCt_bool = fullCt_bool  # データ全体を使うか

        if series_uid:
            self.series_uid = [series_uid]
        else:
            self.series_uid = sorted(getCandidateInfoDict().keys())

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid)

            if self.fullCt_bool:
                self.sample_list += [
                    (series_uid, slice_ndx) for slice_ndx in range(index_count)
                ]
            else:
                self.sample_list += [
                    (series_uid, slice_ndx) for slice_ndx in positive_indexes
                ]

        self.candidateInfoList = getCandidateInfoList()

        series_set = set(self.series_list)
        self.candidateInfo_list = [
            cit for cit in self.candidateInfo_list if cit.series_uid in series_set
        ]

        self.pos_list = [
            nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        log.info(
            "{!r}: {} {} series, {} slices, {} nodules".format(
                self,
                len(self.series_list),
                {None: "general", True: "validation",
                    False: "training"}[isValSet_bool],
                len(self.sample_list),
                len(self.pos_list),
            )
        )

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        ct_t = torch.zeros((self.contextSlices_count*2+1, 512, 512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = start_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.ch_a.shape[0]-1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx

# traing用のデータセット


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 300_000

    def shuffleSample(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)

    def getitem_trainingCrop(self, candidateInfo_tup):
        ct_a, pos_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            (7, 96, 96),
        )
        pos_a = pos_a[3:4]

        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(
            ct_a[:, row_offset: row_offset + 64, col_offset: col_offset + 64]
        ).to(torch.float32)
        pos_t = torch.from_numpy(
            pos_a[:, row_offset: row_offset + 64, col_offset: col_offset + 64]
        ).to(torch.long)

        slice_ndx = center_irc.index

        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx
