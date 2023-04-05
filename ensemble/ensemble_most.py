import os
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import numpy as np

#実行： python /home/kondou/compe2/ensemble/ensemble_most.py


def get_label(result):
    return [int(x)+2 for x in result]

def load_data(openfile):
    with open(openfile, 'r') as f:
        text = f.read().split("\n")
        text = text[:-1] 
    return get_label(text)

COMPE_PATH="/home/kondou/compe2/"

# 出力ファイルまでのPATH
f8 = COMPE_PATH + "8/train8_luke-japanese-large-lite.txt"
f9 = COMPE_PATH + "9/train9_luke-japanese-base-lite.txt"
f10 = COMPE_PATH + "10/train10_luke-japanese-large-lite.txt"
f11 = COMPE_PATH + "11/train11_luke-japanese-large.txt"
f12 = COMPE_PATH + "12/train12_luke-japanese-large-lite.txt"
f13 = COMPE_PATH + "13/train13_luke-japanese-base-lite.txt"
f14 = COMPE_PATH + "14/train14_luke-japanese-large-lite.txt"
f15 = COMPE_PATH + "15/train15_luke-japanese-base-lite.txt"
f16 = COMPE_PATH + "16/train16_luke-japanese-large-lite.txt"
f17 = COMPE_PATH + "17/train17_luke-japanese-large-lite.txt"
f18 = COMPE_PATH + "18/train18_deberta-v2-base-japanese.txt"
f19 = COMPE_PATH + "19/train19_luke-japanese-large-lite.txt"
f20 = COMPE_PATH + "20/train20_luke-japanese-large-lite.txt"
f21 = COMPE_PATH + "21/train21_bert-large-japanese.txt"
f22 = COMPE_PATH + "22/train22_luke-japanese-large-lite_original_data.txt"
f23 = COMPE_PATH + "23/train23_deberta-v2-base-japanese.txt"
f24 = COMPE_PATH + "24/train24_deberta-v2-large-japanese.txt"
f25 = COMPE_PATH + "25/train25_luke-japanese-large-lite_seed222.txt"
f26 = COMPE_PATH + "26/train26_luke-japanese-large_seed222.txt"
f27 = COMPE_PATH + "27/train27_roberta-large-japanese.txt"
f28 = COMPE_PATH + "28/train28_roberta-large-japanese_seed222.txt"
f29 = COMPE_PATH + "29/train29_roberta-large-japanese_seed2023.txt"

# colab分
f30 = COMPE_PATH + "colab/0.541_colab_bert-base-japanese-whole-word-masking.txt"
f31 = COMPE_PATH + "colab/0.555_colab_bert-base-japanese-v2.txt"
f32 = COMPE_PATH + "colab/colab_deberta-v2-base-japanese.txt"
f33 = COMPE_PATH + "colab/colab_waseda_roberta-base-japanese_cosine.txt"
f34 = COMPE_PATH + "colab/colab_waseda_roberta-base-japanese_highlr.txt"
f35 = COMPE_PATH + "colab/colab_waseda_roberta-base-japanese_linear.txt"
f36 = COMPE_PATH + "colab/colab_waseda_roberta-large-japanese_cosine.txt"
f37 = COMPE_PATH + "colab/colab_xlm-roberta-base.txt"
f38 = COMPE_PATH + "colab/colab_xlm-roberta-base_lowlr.txt"
f39 = COMPE_PATH + "colab/colab_xlm-roberta-large.txt"
f40 = COMPE_PATH + "colab/colab_rinna_japanese-roberta-base_cosine.txt"
f41 = COMPE_PATH + "colab/colab_rinna_japanese-roberta-base_linear.txt"

def main():
    # ラベルの読み込み
    res8 = load_data(f8)
    res9 = load_data(f9)
    res10 = load_data(f10)
    res11 = load_data(f11)
    res12 = load_data(f12)
    res13 = load_data(f13)
    res14 = load_data(f14)
    res15 = load_data(f15)
    res16 = load_data(f16)
    res17 = load_data(f17)
    res18 = load_data(f18)
    res19 = load_data(f19)
    res20 = load_data(f20)
    res21 = load_data(f21)
    res22 = load_data(f22)
    res23 = load_data(f23)
    res24 = load_data(f24)
    res25 = load_data(f25)
    res26 = load_data(f26)
    res27 = load_data(f27)
    res28 = load_data(f28)
    # res29 = load_data(f29) #学習中

    res30 = load_data(f30)
    res31 = load_data(f31)
    res32 = load_data(f32)
    res33 = load_data(f33)
    res34 = load_data(f34)
    res35 = load_data(f35)
    res36 = load_data(f36)
    res37 = load_data(f37)
    res38 = load_data(f38)
    res39 = load_data(f39)
    res40 = load_data(f40)
    res41 = load_data(f41)

    # モデル単体のQWKスコア一覧
    # luke-japanese-large-lite 8(0.598), 10(0.598), 16(0.635), 12(0.637), 19(0.642), 14(0.643), 20(0.646), 25(0.646), 17(!0.648), 22_original(0.641)
    # luke-japanese-base-lite 9(0.607), 15(0.621), 13(0.645)
    # luke-japanese-large 11(0.633), 26(0.648)
    # deberta-v2-base-japanese 18(0.606), 32(0.622), 23(0.628)
    # deberta-v2-large-japanese 24(0.631)
    # waseda/roberta-large-japanese 28(0.628), 27(0.633), 36(0.633)
    # waseda_roberta-base-japanese 33(0.596), 34(0.596), 35(0.599)
    # rinna_japanese-roberta-base 41(0.601), 40(0.610)
    # bert-large-japanese 21(0.565)
    # bert-base-japanese-whole-word-masking 30(0.541)
    # bert-base-japanese-v2 31(0.555)
    # xlm-roberta-base 38(0.520), 37(0.593)
    # xlm-roberta-large 39(0.615)
    
    # 出力ファイル名
    OUT_FILE = "ensemble_most_17,26,23,24.txt"

    results = np.array([
        # res25, res26 # # luke-large-lite(seed違い)
        # res25, res20, res21, res24, res27 #largeモデル
        # res26, res17 #0.648と0.648
        # res17, res22, res13, res26, res23, res24, res36, res35, res40, res21, res30, res31, res37, res39 # 全モデル(種類)
        # res17, res22, res13, res26, res23, res24, res36, res40, res39 #0.600越え
        
        # # 順位: 17, 26, 20-25, 13
        # res17, res26, res20 # 1, 1, 3
        # res17, res26, res25 # 1, 1, 3
        # res17, res26, res13 # 1, 1, 4
        # res17, res26, res25, res13 # 1, 1, 3, 4

        # luke-large-lite, luke-large, deberta-v2-base, deberta-v2-large
        # res25, res26, res23, res24
        res17, res26, res23, res24
    ])

    ens = [*map(lambda x: np.argmax(np.bincount(x)), results.T)]
    
    # 出力ファイルへの書き込み
    with open("/home/kondou/compe2/ensemble/"+ OUT_FILE, "w") as f:
        for labeldata in ens:
            f.write(str(int(labeldata)-2))
            f.write("\n")

if __name__ == "__main__":
    main()
