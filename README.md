# PyTorch 実践入門のリポジトリ

<div align="center">
<img src="./etc/表紙.png" alt="PyTorch実践入門" title="PyTorch実践入門" width=50% height=50%>
</div>

書籍「[PyTorch 実践入門](https://www.amazon.co.jp/gp/product/4839974691?pf_rd_r=1GSPXPNMCBQWPW9YW7AW&pf_rd_p=3d55ec74-6376-483a-a5a7-4e247166f80b)」 の日本語版リポジトリです。

[こちら](https://github.com/deep-learning-with-pytorch/dlwpt-code)が原著のリポジトリになります。本リポジトリにあるコードは演習問題の解答をのぞいて全て、上記リポジトリから引用しております。

## 本書で扱う内容

- 第 1 部 PyTorch の基礎
  - 第 1 章 ディープラーニングと PyTorch の概要
  - 第 2 章 訓練済みモデルの利用方法
  - 第 3 章 PyTorch におけるテンソルの扱い方
  - 第 4 章 さまざまなデータを PyTorch テンソルで表現する方法
  - 第 5 章 ディープラーニングの学習メカニズム
  - 第 6 章 ニューラルネットワーク入門
  - 第 7 章 画像分類モデルの構築
  - 第 8 章 畳み込み（Convolution）
- 第 2 部 ディープラーニングの実践プロジェクト：肺がんの早期発見
  - 第 9 章 肺がんの早期発見プロジェクトの解説
  - 第 10 章 LUNA データを PyTorch データセットに変換
  - 第 11 章 結節候補を画像分類するモデルの構築
  - 第 12 章 評価指標とデータ拡張を用いたモデルの改善
  - 第 13 章 セグメンテーションを用いた結節の発見
  - 第 14 章 結節・腫瘍解析システムの全体を構築
- 第 3 部 デプロイメント（Deployment）
  - 第 15 章 本番環境にモデルをデプロイする方法

## 演習問題について

各章の演習問題は該当するフォルダに Jupyter NoteBook 形式で格納されています。
解答の一部は実行の都合上ルートフォルダに配置しています。
実行は基本的に Google Colaboratory 上で行われることを想定しています。
ただし、第 2 部の演習については扱うデータセットのサイズが大きい（約 220GB）ため、Google Coraboratoly の場合はストレージにご注意ください。
また、解答については原著で示されていたわけではなく、日本語版の訳者で解答したものです。

## 疑問点・修正点は Issue にて管理しています。

本 GitHub の Issue にて、疑問点や修正点を管理しています。

不明な点などがございましたら、こちらをご覧ください。

https://github.com/Gin5050/deep-learning-with-pytorch-ja/issues

## 誤植について

書籍中の誤植は以下になります。  
[誤植一覧](https://github.com/Gin5050/deep-learning-with-pytorch-ja/labels/%E8%AA%A4%E6%A4%8D)
