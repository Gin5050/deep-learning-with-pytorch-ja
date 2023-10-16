import random
import copy
import csv
import functools
import glob
import os

from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch10_raw')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open('data/part2/luna/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    candidateInfo_list = []
    with open('data/part2/luna/candidates.csv', "r") as f:
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

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


class Ct:
    """
    識別子を引数に取り, CT画像を読み込み, xyz座標をirc座標に変換し、結節周りの画像を切り出すクラス
    """

    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd),
                        dtype=np.float32)  # [Z, Y, X]

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)  # <- modifies ct_a in place

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        """結節周りの画像を切り出す"""
        center_irc = xyz2irc(  # xyz座標をirc座標に変換
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))  # 切り出しの開始位置
            end_ndx = int(start_ndx + width_irc[axis])  # 切り出しの終了位置

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            # 切り出しの開始位置が画像の範囲外の場合
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            # 切り出しの終了位置が画像の範囲外の場合
            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            # 切り出し
            slice_list.append(slice(start_ndx, end_ndx))

        # 切り出した画像を返す
        ct_chunk = self.hu_a[tuple(slice_list)]

        # 切り出した画像と中心座標を返す
        return ct_chunk, center_irc


def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)  # メモ化することで高速化
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    """
    結節周りの画像を切り出す
    """
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):
    """
    各CTインスタンスに対してPytorchのデータセットに変換するクラス
    """

    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None,):
        self.conadidateInfo_list = copy.copy(
            getCandidateInfoList())  # candidateInfo_listをコピー

        if series_uid:  # series_uidが指定されている場合
            self.conadidateInfo_list = [  # series_uidに一致するものだけに絞る
                x for x in self.conadidateInfo_list if x.series_uid == series_uid]  # xは名前付きタプル

        if isValSet_bool:  # isValSet_boolが指定されている場合(検証データセットの場合)
            assert val_stride > 0, val_stride  # val_strideが0より大きいことを確認
            # val_strideごとにサンプリング
            self.conadidateInfo_list = self.conadidateInfo_list[::val_stride]
            assert self.conadidateInfo_list  # self.conadidateInfo_listが空でないことを確認
        elif val_stride > 0:  # isValSet_boolが指定されていない場合(学習データセットの場合)
            del self.conadidateInfo_list[::val_stride]
            assert self.conadidateInfo_list

        random.shuffle(self.conadidateInfo_list)  # ランダムに並び替え

        log.info("{!r}: {} {} samples".format(  # !rはrepr()を呼び出す
            self,
            len(self.conadidateInfo_list),
            "validation" if isValSet_bool else "training",
        ))

    def __len__(self):
        return len(self.conadidateInfo_list)

    def __getitem__(self, ndx):
        # candidateInfo_listからndx番目の要素を取り出す
        candidateInfo_tup = self.conadidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)  # チャンネル次元を追加

        pos_t = torch.tensor([  # 結節の有無を表すラベルを作成
            # 結節ならtorch.tensor[False, True] -> [0, 1], 結節でないならtorch.tensor[True, False] -> [1, 0]
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
        ], dtype=torch.long)

        return (candidate_t, pos_t, candidateInfo_tup.series_uid, torch.tensor(center_irc))
