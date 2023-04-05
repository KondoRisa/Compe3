import torch
from transformers import BertJapaneseTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, SequentialSampler, RandomSampler
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import os
import string
from sklearn.metrics import cohen_kappa_score
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments,
    LukeTokenizer, LukeForSequenceClassification,
    pipeline, EarlyStoppingCallback
)

# 実行は：python /home/kondou/compe2/17/train17.py > /home/kondou/compe2/17/1_luke_train.log

DATA_PATH = "/home/kondou/compe2/data/self_data2/" # dataディレクトリまでのpath
OUT_FNAME = "train17_luke-japanese-large-lite.txt" # 出力先ディレクトリ

# MODEL_NAME = "studio-ousia/luke-japanese-base"
# MODEL_NAME = "studio-ousia/luke-japanese-base-lite"
# MODEL_NAME = "studio-ousia/luke-japanese-large"
MODEL_NAME = "studio-ousia/luke-japanese-large-lite"

class MyDataset(Dataset):
    """トークン入力データセット"""
    def __init__(self, encodings, labels=None):
      self.encodings = encodings
      self.labels = labels

    def __len__(self):
      return len(self.encodings['input_ids'])

    def __getitem__(self, index):
      # input = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
      input = {key: val[index].clone().detach() for key, val in self.encodings.items()}
      
      if self.labels is not None:
          input["label"] = torch.tensor(self.labels[index])

      return input

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_data(openfile):
    with open(openfile, 'r') as f:
        text = f.read().split("\n")
        text = text[:-1] # train_textの最後に''があるため削除
    return text

def compute_metrics(pred):
    """メトリクス定義"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    #2値分類ならaverage='binary'とする
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')
    print("正解：", labels)
    print("推論：", preds)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'QWK': qwk,
    }


def main():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    #SEEDの固定
    SEED = 42
    seed_everything(SEED)


    # データの読み込み
    train_text = load_data( DATA_PATH + "text.train.txt")
    dev_text = load_data( DATA_PATH + "text.dev.txt") # 検証用
    test_text = load_data( DATA_PATH + "text.test.txt") # 提出用

    # ラベルの読み込み
    train_label = load_data( DATA_PATH + "label.train.txt")
    dev_label = load_data( DATA_PATH + "label.dev.txt")

    y_train = [x+2 for x in list(map (int, train_label))] #ラベルを0始まりに
    y_dev = [x+2 for x in list(map (int, dev_label))]

    # 後で間違えたものを確認するためのDataFrame
    valid = pd.DataFrame(columns=['label', 'sentence']) 
    for label, text in zip(y_dev, dev_text):
        valid = valid.append({'label':label , 'sentence':text}, ignore_index=True) #labelはint

    # 顔文字の除去
    # table = str.maketrans("", "", string.punctuation + "「」、。〇♥♣♦・∀゚≡#\"/ ● .♡ \〜δ_ | ‾ | ○・ w+ W+ (   _ д _ ) 「  」 … 『 』（  ）→@ &δ ()【  】( ゜ Ω ゜ )  ? < * @  >(^^)  ”  ヽ ( * ^ ω ^ * )  (´・ω・`)(^_^) ( ●   ́ω ` ● )  _ o _   ) / (・・; ) (^ω^) ♪ ( ●   ́з ` ● )  (^▽^) ?(   ́д ` | | | )( ( ( ( 。 д ゜ ) ) )( ゜Д゜)(´∇`)( ´ ∇ ` * )( ・ _ ・ ` ) (>_<) - ( ´ - ε - ` )%  (   _ ・ ω ・ _   )(・ω・)゛‼×(*_*)( ; ´ _ ゝ ` )(T_T) :  / / ∇ / / ) (⌒▽⌒)(*´∀`*)☆ * : . 。 .   o ( ≧ ▽ ≦ ) o   . 。 . : * ☆ (´Д`)\ (・o・) /( ^q^ ) ( - ω - ) \(^O^)/ (   ;   ゜ д ゜ ) ( ‾ ▽ ‾ ; )(´Д`)(´Д⊂ )  ( ´ - ω - ` )( ; ω ; )( ; ゛゜ ＇ ω ゜ ＇ ) :  _ ( : 3   」 ∠ ) _ (  ; ) (   ́д `   ) ( ‾ ▽ ‾ ; )( ´ ; д ; ` ) \ (   ＇ ω ＇ ) / ( 「   ゜ д ゜ ) ( *   ́ェ ` * ) ・ : * + . (   ° ω °   ) ) / . : +  ( : 3 [ _ _ _ ]  (       。 > _ < 。 )     ♡   ♡  ( _ ω _ )(^_^;)「 ヾ ( ⌒ ( ノ ＇ ω ＇ ) (^ω^)( o ´  ` )( ´ - ` ) ° ∀ °   )_ ( _ ` ~ ´ _ ) _. ∧ _ ∧       ∧ _ ∧ . (     ` ・ ω ・ )  ) ゜ д ゜ ) ・ ゜ (          r ⊂   ⊂ ) ＇ |     _    ⊂ _ ⊂ ノ `  ´  (    ́ω ` c ): ; ( ∩ ´ _ ` ∩ ) ; :  _ : ( ´ _ ` 」   ∠ ) :  ( ´ ∀ ` = ) (   ＇ ч ＇   ) ♪ _ _   (   _ ω _   ξ   _ ω _   )   _ _ (  °   ́o ` ° c ) □ ♡ ▽ ○☆   _ o _ ) /_ : ( ´ - ` 」   ∠ ) : _  ( ‘ 、 3 _ ヽ ) _( _ ´ ÷ ` _   )( _ ´ ÷ ` _   ) ( ゜ ` ω ´   ゜ ) ゜ 。 ゜ ( ゜ ´ 〜 ` ゜ ) ゜ 。[ - ━ - ] Ω [ - ━ - ] _ (   : 0   」   ) _ ( _ ´ ¬ ` _   )( _ 3 _ ヽ ) _[ _ _ ] ; * д * ) _  (   − ω −   ) ( >< 。 )[ _ _ ] ε _ )   〜 (   ε _ )   0 ( ´ × ω × ` ) ( /   ́a ` )( ( * _ ¬ _ * ) ) (   _ _ _ _   ≡   _ _ _ _   )          (~(~))~))           ( | | / / /              | / / /         / ‾ ‾       /   _ ω _     |       |       . : /       |     . : /       _ l _   . : /     (   _ _ ノノ     |   )   | _ 2 ⌒ )     | /         \ _ )   ( 2 _ )( * N   ́ω ` n * )  ( ( o ( 。 > Ω < 。 ) o ) )  _ (   * _ 0 _ * ) _         ⊂ (   ? o ?   ) ⊃        ( _ _   _ ω _   ) _ _ _ _ ♡  ∩ ^ ω ^ ∩   σ ( _ _ 口 _ _ ) ! !   ( ´ - ` ) . 。       ( ´ - д - ) - 3    (   ^ ω ^   ) (   ゜ ∀ ゜ )  彡 ゜  ┏(   ^  ^)┛ ( ‾  ‾ ) (*´艸`*) ♡( ( ( о ( * ° ▽ ° * ) ο ) ) )     ) ゜ д ゜ ( ヽ( # ^ ω ^ )(   _ ・ ω ・ _   ) . 。 _  (   ́φ _ φ ` ) ヽ ( ° ∀ 。 )  ( ゜ ∀ ゜ ( > 〈 < 。 ) / 。 ゜ ( ゜   ́д ` ゜ ) ゜ 。ヽ ( * ゜ ω ゜ )ε - ( ´ - ` * )( ノ * ° ▽ ° ) ( ` - ω - ´ ) _ ( * ´ ) ` * ) _( . _ . )( ´ ・ ω ・ ` ; ) _ ( - Ω -`_)⌒)_ ( * ´ _ ` * ) ヽ ( ´ ▽ ` ) /( ;  ; ) (;_;)( ‾ ▽ ‾ ) b (   ́ω ` )( ‾ 〜 ‾ ; )(´;ω;`) ( ^  ^ ; ) ((+_+)) ( 。 ・ ω ・ 。 ) ( ´_ゝ`)( ; ^ ω ^ ) 。 ・ ゜ ・ (ノ∀`) ・ ゜ ・ 。( ´ - ` ) . 。  (_ ( ┐ 「 ε : ) _""")
    # train_text_demoji = [t.translate(table) for t in train_text] #... ・・・　…　あり/なし
    # test_text_demoji = [t.translate(table) for t in test_text]
    # dev_text_demoji = [t.translate(table) for t in dev_text]
    train_text_demoji = train_text
    dev_text_demoji = dev_text
    test_text_demoji = test_text

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenizeのテスト
    ids = tokenizer.encode(train_text_demoji[0])
    wakati = tokenizer.convert_ids_to_tokens(ids)
    print(train_text_demoji[0]) #原文
    print(ids) #トークンID
    print(wakati) #IDを戻すと

    # モデルの選択 : num_labelsラベルの種類数を指定
    model = LukeForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

    # dataset作成
    text_lengths_list = [len(tokenizer.encode(text)) for text in train_text_demoji]
    train_X = tokenizer(train_text_demoji, return_tensors='pt', padding="max_length", max_length=max(text_lengths_list), truncation=True)
    valid_X = tokenizer(dev_text_demoji, return_tensors='pt', padding="max_length", max_length=max(text_lengths_list), truncation=True)
    test_X = tokenizer(test_text_demoji, return_tensors='pt', padding="max_length", max_length=max(text_lengths_list), truncation=True)

    # データの確認
    print(test_X)
    print(f"最大トークン数は{max(text_lengths_list)}です")
    print(train_X.keys())

    train_ds = MyDataset(train_X, y_train)
    valid_ds = MyDataset(valid_X, y_dev)
    test_ds = MyDataset(test_X)

    print(train_ds[0].keys())
    print(f"学習データ数は{len(train_ds)}です")
    print(train_ds[0]["input_ids"])
    print(tokenizer.decode(train_ds[0]["input_ids"])) 



    # Trainerの設定
    train_args = TrainingArguments(
        output_dir                  = "./out", #log出力場所
        overwrite_output_dir        = True, #logを上書きするか
        load_best_model_at_end      = True, #EarlyStoppingを使用するならTrue
        metric_for_best_model       = "QWK", #EarlyStoppingの判断基準。7-1. compute_metricsのものを指定
        save_total_limit            = 1, #output_dirに残すチェックポイントの数
        save_strategy               = "epoch", #いつ保存するか？
        evaluation_strategy         = "epoch", #いつ評価するか？
        logging_strategy            = "epoch", #いつLOGに残すか？
        label_names                 = ['labels'], #分類ラベルのkey名称(デフォルトはlabelsなので注意)
        lr_scheduler_type           = "linear", #学習率の減衰設定(デフォルトlinearなので設定不要)
        learning_rate               = 5e-7, #学習率(デフォルトは5e-5)
        num_train_epochs            = 200, #epoch数
        per_device_train_batch_size = 32, #学習のバッチサイズ
        per_device_eval_batch_size  = 32, #バリデーション/テストのバッチサイズ
        seed                        = SEED, #seed
    )

    MyCallback = EarlyStoppingCallback(early_stopping_patience=10)

    #Trainerを定義
    trainer = Trainer(
        model=model, #モデル
        args=train_args, #TrainingArguments
        tokenizer=tokenizer, #tokenizer
        train_dataset=train_ds, #学習データセット
        eval_dataset=valid_ds, #validデータセット
        compute_metrics = compute_metrics, #compute_metrics
        callbacks=[MyCallback] #callback
    )

    #学習
    trainer.train()

    #valid評価
    valid_preds = trainer.predict(valid_ds)

    #元のvalidデータフレームにpredカラムを追記する
    valid['pred'] = np.argmax(valid_preds.predictions, axis=1)

    # valid上で誤検知しているものを表示
    print(valid[valid['label']!=valid['pred']])

    # print("validに対するQWK", cohen_kappa_score(valid['label'], valid['pred'], weights='quadratic'))


    # testに対して推論
    test_preds = trainer.predict(test_ds)
    preds = np.argmax(test_preds.predictions, axis=1)

    # モデルの保存
    trainer.save_model("/home/kondou/compe2/17/model") # os.path.join(args.save_model_dir, f"fold{fold_id}"))

    # testに対する推論結果を書き込み
    with open("/home/kondou/compe2/17/"+OUT_FNAME, "w") as f:
        # print(preds)
        for pred in preds:
            # print(pred)
            f.write(str(int(pred)-2) + "\n")


if __name__ == "__main__":
    main()