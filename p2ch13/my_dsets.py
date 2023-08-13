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
    # 結節か, アノテーションがあるか, 悪性か, 直径, シリーズUID(患者のID), 中心座標
    'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz'
)

@functools.lru_cache(1) # メモ化
def getCandidateInfoList(requireOnDisk_bool=True):
    """
    [(結節か, アノテーションがあるか, 悪性か, 直径, シリーズUID(患者のID), 中心座標), (...), ...]を返す
    """
    mhd_list = glob.glob('data-unversioned/part2/luna/sublset*/*.mhd') # ファイルのリスト
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list} # os.path.split(p)はパスをディレクトリとファイル名に分け, [-1]はファイル名のみを取得, [:-4]は拡張子を除外
    
    candidateInfo_list = []

    with open('data/part2/luna/annotations_with_maliganancy.csv', 'r') as f: # 結節のファイルはここですべて網羅されてる
        for row in list(csv.reader(f))[1:]: # csv.reader(f)はcsvファイルを読み込み, list(csv.reader(f))はリストに変換, [1:]はヘッダーを除外
            series_uid = row[0] # 患者(CT画像)のID
            if series_uid not in presentOnDisk_set and requireOnDisk_bool: # 参照したseries_uidに対してmhdファイルがなければスキップ
                continue
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]]) # アノテーションの中心座標
            annotationDiameter_mm = float(row[4]) # アノテーションの直径
            isMal_bool = {'False': False, 'True': True}[row[5]] # 悪性かどうか

            candidateInfo_list.append(
                CandidateInfoTuple(
                True,
                True,
                isMal_bool,
                annotationDiameter_mm,
                series_uid,
                annotationCenter_xyz
                )
            )
    
    with open('data/part2/luna/candidate.csv', 'r') as f: 
        # annotationファイルが改良されて結節はすべてcandidateInfo_listに含まれているため, ここでは結節にみえても結節ではないもののみを追加する
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4])) # 1の場合はannotationファイルで取得済み
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = 0.0 # 結節でない場合しか考えないため0

            candidateInfo_list.append(
                CandidateInfoTuple(
                    False,
                    False,
                    False,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz
                )
            )
    
    candidateInfo_list.sort(reverse=True) # 直径の大きい順にソート
    return candidateInfo_list

@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    """
    {シリーズUID(患者のID): [(結節か, アノテーションがあるか, 悪性か, 直径, シリーズUID(患者のID), 中心座標), (...), ...], ...}を返す
    """
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        # uidをキーとしてcandidateInfo_dictにcandidateInfo_listの要素(タプル)を追加
        # すでにキーが存在する場合はそのキーに対応するリストに値を追加し, 存在しない場合は空の配列を用意して値を追加
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid, []).append( 
            candidateInfo_list
        )
    
    return candidateInfo_dict

class Ct:
    """
    与えられたシリーズUIDに対応するCT画像に関する情報を保持する
    """
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data-unvesioned/part2/luna/subset*/{}.mhd'.format*(series_uid)
        )[0]
        
        ct_mhd = sitk.ReadImage(mhd_path) # SimpleSTKモジュールを用いてmhdファイルを読み込む
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32) # CT画像のボクセルごとの値をnumpy配列として取得

        self.series_uid = series_uid
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin()) # CT画像の原点(xyz座標)
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing()) # CT画像のボクセルの大きさ(xyz座標, 医療データはボクセルの大きさが立方体でないことがある)
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3) # CT画像のボクセルの向き(3x3の行列)

        candidateInfo_list = getCandidateInfoDict()[series_uid] # 与えられたシリーズUIDに対応する結節のリストを取得. 一つのCT画像に対して複数の結節候補があるかもしれない

        self.positiveInfo_list = [ # 結節候補のリストから結節のみを取り出す
            candidate_tup for candidate_tup in candidateInfo_list if candidateInfo_list.isNodule_bool
        ] 

        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list) # 結節のアノテーションをマスクとして取得
        self.positive_indexes = (
            self.positive_mask.sum(axis=(1, 2).nonzero()[0]) # 結節のアノテーションのあるボクセルのインデックスを取得
        )

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu=-700):
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool) # CT画像のボクセルごとの値をbool型のnumpy配列として初期化

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
                while self.hu_a[ci+index_radius, cr, cc] > threshold_hu and self.hu_a[ci-index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[ci, cr+row_radius, cc] > threshold_hu and self.hu_a[ci, cr-row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1
            
            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc+col_radius] > threshold_hu and self.hu_a[ci, cr, cc-col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1
        
            boundingBox_a[
                ci-index_radius: ci+index_radius+1,
                cr-row_radius: cr+row_radius+1,
                cc-col_radius: cc+col_radius+1
            ] = True
        
        mask_a = boundingBox_a & (self.hu_a > threshold_hu) # 角を取る

        return mask_a

    def getRawCandidate(self, center_xyz, width_irc): # 結節候補周辺のボクセルを取得
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(round(start_ndx + width_irc[axis])) 

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr( # 切り出すボクセル集合の中心がCT画像の範囲内にあるか確認
                [
                    self.series_uid,
                    center_xyz,
                    self.origin_xyz,
                    self.vxSize_xyz,
                    center_irc,
                    axis
                ]
            )

            # 切り出すボクセル集合の始点がCT画像の範囲外にある場合
            if start_ndx < 0: 
                start_ndx = 0
                end_ndx = int(width_irc[axis])
            
            # 切り出すボクセル集合の終点がCT画像の範囲外にある場合
            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))
        
        ct_chunk = self.hu_a[tuple(slice_list)] # 実際の密度データが入った3D配列から切り出す
        pos_chunk = self.positive_mask[tuple(slice_list)] # 結節のアノテーションが入った3D配列から切り出す

        return ct_chunk, pos_chunk, center_irc
 
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc

@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid): # CT画像の深さ(スライス数)と結節のアノテーションがTrueであるスライスのインデックスを返す関数
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes

class Luna2dSegmentationDataSet(Dataset):
    """segmentationモデルに入力するデータセットを作るクラス(訓練用データセットを作るクラスはこのクラスを継承する)"""
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None, contextSlices=3, fullCt_bool=False):
        self.contextSlices = contextSlices
        self.fullCt_bool = fullCt_bool

        if series_uid: # 一つのCT画像のみを扱う場合
            self.series_list = [series_uid]
        else:
            self.series_list = copy.copy(getCandidateInfoDict().keys())
        
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list =  self.series_list[::val_stride]
            assert self.series_list # self.series_listが空でないことを確認
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid) # CT画像の深さ(スライス数), 結節のアノテーションがTrueであるスライスのインデックスを取得

            if self.fullCt_bool: # CT画像内のすべてのスライスを用いる場合
                self.sample_list += [
                    (series_uid, slice_ndx) for slice_ndx in range(index_count)
                ]
            else: # Trueを含むスライスのみを用いる場合
                self.sample_list += [
                    (series_uid, slice_ndx) for slice_ndx in positive_indexes
                ]
            
        self.candidateInfo_list = getCandidateInfoList()

        series_set = set(self.series_list) # 重複はないはずだけど...
        self.candidateInfo_list = [
            x for x in self.candidateInfo_list if x.series_uid in series_set
        ]

        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool] # 結節であるcandidateInfoのみを取り出し、リストにする

        log.info('{!r}: {} {} series, {} slices, {} nodules'.format(
            self,
            len(self.series_list), # 今回見るCT画像の数
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool], # 今回のモード(訓練か検証かそれ以外か)
            len(self.sample_list), # ((uid, slice_ndx), (uid, slice_ndx), ...)のような形式
            len(self.pos_list) # 結節の数
        ))
    
    def __len__(self): # たくさんあるデータセットの長さを返す関数(必須)
        return len(self.sample_list)
    
    def __getitem__(self, ndx): # たくさんあるデータセットに対してndxを指定するとそのインデックスのモデルへの入力データとラベルを返す関数(必須)
        # 剰余を取ることでデータセットのインデックスがデータセットの長さを超えた場合にも対応できる(もしかしたらサンプル数を超えたインデックスにデータセットがアクセスするかもしれない)
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)
    
    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        ct_t = torch.zoros((self.contextSlice_count*2+1, 512, 512)) # 今回はcontextSlice_count=3なので7スライス分のデータを用意する(手前3スライス, 今回のスライス, 奥3スライス)

        start_ndx = slice_ndx - self.contextSlices
        end_ndx = slice_ndx + self.contextSlices + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0) # スライスのインデックスが0未満にならないようにする
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1) # スライスのインデックスがCT画像の深さを超えないようにする
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))
        
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0) # 実際の結節のアノテーション(答え), 形状は(1, 512, 512)

        return ct_t, pos_t, ct.series_uid, slice_ndx # モデルへの入力データ, ラベル, CT画像のID, 入力データのスライスのインデックス, 最後2つはデバッグ用で、訓練には使わない
    

class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataSet):
    """
    訓練用のデータセットを作るクラス
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2 # 

        def __len__(self):
            return 300000
        
        def shuffleSamples(self):
            random.shuffle(self.candidateInfo_list) # 全ての結節候補をシャッフル
            random.shuffle(self.pos_list) # 結節のみをシャッフル

        def __getitem__(self, ndx):
            candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
            return self.getitem_trainingCrop(candidateInfo_tup)
        
        def getitem_trainingCrop(self, candidateInfo_tup):
            ct_a, pos_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid, 
                candidateInfo_tup.center_xyz,
                (7, 96, 96),
            )

            pos_a = pos_a[3:4] # 今回見ているスライスのアノテーション

            row_offset = random.randrange(0, 32)
            col_offset = random.randrange(0, 32)

            ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset+64, col_offset:col_offset+64]).to(torch.float32) # ランダムに64x64の領域を切り出し、7x64x64のテンソルに変換
            pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset+64, col_offset:col_offset+64]).to(torch.long)

            slice_ndx = center_irc.index

            return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx
        