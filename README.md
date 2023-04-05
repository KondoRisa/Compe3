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
<br>

## 使用したモデル
|  モデル  |  Validに対するQWK  | 提出データに対するQWK |
| ---- | ---- | ---- |
|  luke-japanese-large-lite  |  0.6312494709842178  |  0.648  |
|  luke-japanese-base-lite  |  0.6080105677221286  |  0.645  |
|  luke-japanese-large  |  0.6310931095474535  |  0.648  |
|  deberta-v2-base-japanese  |  0.5956303347946228  |  0.628  |
|  deberta-v2-large-japanese  |  0.6232196817019955  |  0.631  |
|  waseda/roberta-base-japanese  |  --35  |  0.599  |
|  waseda/roberta-large-japanese  |  --36  |  0.633  |
|  rinna_japanese-roberta-base  |  --40  |  0.610  |
|  xlm-roberta-base  |  --37  |  0.593  |
|  xlm-roberta-large  |  --39  |  0.615  |
|  cl-tohoku/bert-base-japanese-v2  |  --31  |  0.555  |
|  cl-tohoku/bert-large-japanese  |  0.5572011420607461  |  0.565  |
|  cl-tohoku/bert-base-japanese-whole-word-masking  |  --30  |  0.541  |

luke-japanese-large-lite 根性マイニング前だと0.641
<br>



