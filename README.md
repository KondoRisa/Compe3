# 第３回 研究室卒業論文テーマ決めコンペ

- タスク  <br>
日本語のTwitterテキストの感情極性を５クラスに分類 （-2, -1, 0, 1, 2)　<br>
評価指標は Quadratic Weighted Kappa <br>

- 制約 <br>
ニューラルネットワーク・外部データを使用しない <br>
データセットの訓練用・検証用・提出用の分割を変更しない <br>
使用するGPUは１人１枚まで

- データセットは[WRIME](https://github.com/ids-cv/wrime) <br>
訓練用：30,000 件　<br>
検証用：2,500 件　<br>
提出用：2,500 件　<br>
<br>


## 使用したモデル
|  モデル  |  Validに対するQWK  | 提出データに対するQWK |
| ---- | ---- | ---- |
|  luke-japanese-large-lite  |  TD  |  0.648  |
|  luke-japanese-base-lite  |  TD  |  0.645  |
|  luke-japanese-large  |  TD  |  0.648  |
|  deberta-v2-base-japanese  |  TD  |  0.628  |
|  deberta-v2-large-japanese  |  TD  |  0.631  |
|  waseda/roberta-base-japanese  |  TD  |  0.599  |
|  waseda/roberta-large-japanese  |  TD  |  0.633  |
|  rinna_japanese-roberta-base  |  TD  |  0.610  |
|  xlm-roberta-base  |  TD  |  0.593  |
|  xlm-roberta-large  |  TD  |  0.615  |
|  cl-tohoku/bert-base-japanese-v2  |  TD  |  0.555  |
|  cl-tohoku/bert-large-japanese  |  TD  |  0.565  |
|  cl-tohoku/bert-base-japanese-whole-word-masking  |  TD  |  0.541  |
<br>
luke-japanese-large-lite 根性マイニング前だと0.641
<br>


## 実行手順
1. 入力として、与えられたデートセットを根性マイニングで整えたものを使用した
1. 数字のついた各フォルダ内にある`train〇(数字).py`を
  ```
  python ~/〇(数字)/train〇(数字).py > ~/〇(数字)/logのファイル名
  ```
  と実行した。
1. アンサンブルは、ensembleフォルダにある`ensemble.py`（足し合わせて四捨五入をすることでアンサンブル）と`ensemble_most.py`（多数決をすることでアンサンブル）を用いた
  ```
  python ~/ensemble/ensemble.py
  ```
  `ensemble_most.py`（多数決をすることでアンサンブル）よりも`ensemble.py`（足し合わせて四捨五入をすることでアンサンブル）の結果の方が全体を通して高かった。
