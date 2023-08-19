import argparse
import datetime
import os
import sys

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import LunaDataset
from util.logconf import logging
from .model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a.
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]  # sys.argvが指定されなかったら, コマンドライン引数をそのまま使う

        # argparse.ArgumentParserとは, コマンドライン引数を解析するためのパーサー
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )

        parser.add_argument('--tb-prefix',
                            default='p2ch11',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='dwlpt',
                            )
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime(
            '%Y-%m-%d_%H.%M.%S')  # タイムスタンプを取得

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(
                torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:  # GPUが複数ある場合は, マルチGPUを使う
                # マルチGPUを使う場合は, モデルをnn.DataParallelでラップする
                model = nn.DataParallel(model)
            model = model.to(self.device)  # モデルのパラメータをGPUに転送
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        # return Adam(self.model.parameters())

    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()  # バッチサイズをGPU数でかける

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,  # データローダーの並列処理数
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTensorboardWriters(self):  # 何を保存しているの? ログ?
        if self.trn_writer is None:
            log_dir = os.path.join(  # ログの保存先
                'runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(  # TensorBoardのwriterにより, ログを保存
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)  # 訓練したあとに, ログを保存

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()  # モデルを訓練モードにする
        trnMetrics_g = torch.zeros(  # 訓練データの評価指標を格納するテンソル
            METRICS_SIZE,  # METRICS_SIZE = 3
            len(train_dl.dataset),  # 全データの数
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(  # dataloaderを直接使うのではなく, enumerateWitdEstimateを使うことで, バッチの進捗を表示できる
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

            # # This is for adding the model graph to TensorBoard.
            # if epoch_ndx == 1 and batch_ndx == 0:
            #     with torch.no_grad():
            #         model = LunaModel()
            #         self.trn_writer.add_graph(model, batch_tup[0], verbose=True)
            #         self.trn_writer.close()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():  # 勾配を計算しない(処理が高速に行われる)
            self.model.eval()  # モデルを評価モードにする
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(  # dataloaderを直接使うのではなく, enumerateWitdEstimateを使うことで, バッチの進捗を表示できる
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            # batch_ndx: バッチのインデックス, batch_tup: バッチの要素(タプル)
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(  # 損失を計算
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        # バッチのタプルの分解 (input_t: 結節テンソルのバッチ, label_t: 0, 1のバッチ)
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        # logit(softmaxかませる前)とprobability(softmax噛ませたやつ)を計算
        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g,
            label_g[:, 1],
        )
        start_ndx = batch_ndx * batch_size  # 何個目のバッチか * バッチサイズ
        end_ndx = start_ndx + label_t.size(0)  # バッチの終了時のデータのインデックス

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:, 1].detach()  # 0, 1のラベル
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:, 1].detach()  # 結節である確率
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()  # 損失(バッチごとではなくサンプルごとに計算されている)

        return loss_g.mean()  # 全データ(サンプル)で損失を平均

    def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        # 0.5以下のラベルは陰性
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        # 0.5以下の予測は陰性
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask  # 陰性のマスクを反転
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())  # 陰性の数(実際のラベル)
        pos_count = int(posLabel_mask.sum())

        # 陰性のうち, 陰性と予測できた数
        neg_correct = int((negLabel_mask & negPred_mask).sum())
        # 陽性のうち, 陽性と予測できた数
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        metrics_dict = {}
        metrics_dict['loss/all'] = \
            metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = \
            metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()  # 陰性の損失
        metrics_dict['loss/pos'] = \
            metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()  # 陽性の損失

        metrics_dict['correct/all'] = (pos_correct + neg_correct) \
            / np.float32(metrics_t.shape[1]) * 100  # 全体の正解率
        metrics_dict['correct/neg'] = neg_correct / \
            np.float32(neg_count) * 100  # 陰性の正解率
        metrics_dict['correct/pos'] = pos_correct / \
            np.float32(pos_count) * 100  # 陽性の正解率

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x/50.0 for x in range(51)]

        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

        # score = 1 \
        #     + metrics_dict['pr/f1_score'] \
        #     - metrics_dict['loss/mal'] * 0.01 \
        #     - metrics_dict['loss/all'] * 0.0001
        #
        # return score

    # def logModelMetrics(self, model):
    #     writer = getattr(self, 'trn_writer')
    #
    #     model = getattr(model, 'module', model)
    #
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             min_data = float(param.data.min())
    #             max_data = float(param.data.max())
    #             max_extent = max(abs(min_data), abs(max_data))
    #
    #             # bins = [x/50*max_extent for x in range(-50, 51)]
    #
    #             try:
    #                 writer.add_histogram(
    #                     name.rsplit('.', 1)[-1] + '/' + name,
    #                     param.data.cpu().numpy(),
    #                     # metrics_a[METRICS_PRED_NDX, negHist_mask],
    #                     self.totalTrainingSamples_count,
    #                     # bins=bins,
    #                 )
    #             except Exception as e:
    #                 log.error([min_data, max_data])
    #                 raise


if __name__ == '__main__':
    LunaTrainingApp().main()
