import argparse
import datetime
import os
import sys

import numpy as np

from torcn.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import LunaDataset
from util.logconf import logging
from .model import LunaModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

METRICS_LABEL_NDX = 0
METRICS_PRED_NEX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class LunaTrainingAPP:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            # sys_argvはリスト型, ファイルをコマンド引数をつけて実行した時に、ファイル名が0番目に入るので、1番目以降の引数を取得する
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParseer()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int)
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int)
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int)
        parser.add_argument('--tb-prefix',
                            default='p2ch11',
                            help='Data prefix to use for Tensorboard run. Defaults to chapter.')
        parser.add_argument('comment',
                            help='Comment suffix for Tensorboard run.',
                            nargs='?',
                            default='dlwpt')
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')  # 現在時刻を取得

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()  # bool
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info('Using CUDA; {} device.'.format(
                torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        # return Adam(self.model.parameters())

    def initTrainDl(self):
        train_ds = LunaDataset(val_stride=10, isValSet_bool=False)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,  # 並列処理を行うプロセス数
            pin_memory=self.use_cuda  # メモリピン留めを行うかどうか, ページングを防ぎ処理を高速化する
        )

        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(val_stride=10, isValSet_bool=True)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )

    # Tensorboardに保存する準備(保存先のPathの設定, writerの初期化)
    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join(
                'runs', self.cli_args.tb_prefix, self.time_str)  # 保存先のディレクトリ

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)  # トレーニングデータの保存先
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '=val_cls-' + self.cli_args.comment)  # バリデーションデータの保存先

    def main(self):
        log.info('Starting {}, {}'.format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs+1):
            log.info('Epoch {} of {}, {}/{} batches of size {}*{}'.format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1)
            ))

            trnMetrics_t = self.doTraining(
                epoch_ndx, train_dl)  # トレーニングデータの学習, 評価指標の計算
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        if hasattr(self, 'trn_writer'):  # 訓練を一通り終えるとtrn_writerが存在する(Noneじゃなくなる)ので、Tensorboardを閉じる
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()  # trainモードに変更
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,  # 3
            len(train_dl.dataset),  # 全データ数
            device=self.device
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            'E{} Training'.format(epoch_ndx),
            start_ndx=train_dl.num_workers
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                'E{} Validation'.format(epoch_ndx),
                start_ndx=val_dl.num_workers
            )

            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g, label_g[:, 1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX,
                  start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NEX,
                  start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t, classificationThreshold=0.5):
        self.initTensorboardWriters()
        log.info("E{} {}".format(epoch_ndx, type(self).__name__))

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NEX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())  # TNの数
        pos_correct = int((posLabel_mask & posPred_mask).sum())  # TPの数

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX,
                                             negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX,
                                             posLabel_mask].mean()  # 陽性の損失

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / \
            np.float32(metrics_t.shape[1]) * 100  # 全体の正解率
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

        # self.trn_writer or self.val_writerを取得
        writer = getattr(self, mode_str + '_writer')
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        # PR(Precision-Recall)曲線の描画
        writer.add_pr_curve('pr', metrics_t[METRICS_LABEL_NDX],
                            metrics_t[METRICS_PRED_NEX], self.totalTrainingSamples_count)

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
    # Pythonスクリプトとして実行された時に、__name__は'__main__'になる
    LunaTrainingAPP().main()
