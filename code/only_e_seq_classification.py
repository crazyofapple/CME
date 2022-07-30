#! -*- coding:utf-8 -*-

import numpy as np
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense, Dropout
import time
import json

set_gelu('tanh')  # 切换gelu版本

num_classes = 5
maxlen = 512
batch_size = 8
config_path = '/raid/$Anonymous$/$Anonymous$/bert_pretrained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = '/raid/$Anonymous$/$Anonymous$/bert_pretrained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = '/raid/$Anonymous$/$Anonymous$/bert_pretrained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'
model_name = "best_model.weights.only.expls.sequence.classification"

    

def trans(ch):
    if ch == "A":
        return 0
    elif ch == "B":
        return 1
    elif ch == "C":
        return 2
    elif ch == "D":
        return 3
    elif ch == "E":
        return 4

def load_data(filename, type=None):
     
    D = []
    with open(filename, "r") as f:
        r_data = json.load(f)
        for ind, r in enumerate(r_data): 
            r['question'] = r['question'].replace(" ", "").replace("“", "\"").replace("”", "\"")
            r['explanation'] = r['explanation'].replace(" ", "").replace("“", "\"").replace("”", "\"")
            r['options'] = [x.strip().replace("“", "\"").replace("”", "\"") for x in r['options']]
            options_text = ""
            abc = ["A.", "B.", "C.", "D.", "E."]
            for ch, option in zip(abc, r['options']):
                options_text += (ch + option)
        
            if type is not None:
                # f"此题的答案是:{r['options'][trans(r['answer'])]}。解析:"
                # generated_expl[ind] = r['explanation']
                # "此题的解析是"
                if type == "xe":
                    generated_expl[ind] = generated_expl[ind].replace("此题的解析是", "")
                    text = '{}？选项是:{}。参考文本为:{}。'.format(r['question'], options_text, generated_expl[ind]) #r['explanation']) # cos-e
                elif type == "e":
                    generated_expl[ind] = generated_expl[ind].replace("此题的解析是", "")
                    text = '参考文本为:{}。'.format(generated_expl[ind]) #, r['explanation']) # cos-e
                elif type == "x":
                    text = '{}？选项是:{}。'.format(r['question'], options_text) # cos-e
            else:
                # text = '{}？选项是:{}。参考:{}。'.format(r['question'], options_text, context_str) # cos-e
                # text = '{}？选项是:{}。'.format(r['question'], options_text) # cos-e
                text = '{}。'.format(r['explanation'].strip("。"))
            label = trans(r['answer'])
            D.append((text, int(label)))
            print(D[-1])
    return D


# 加载数据集
train_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/train_context.json')
valid_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/test_context.json')
test_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/test_context.json')
# train_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v1.9_data/train_context_v0_only_q.json')
# valid_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v1.9_data/test_context_v0_only_q.json')
# test_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v1.9_data/test_context_v0_only_q.json')
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)



class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='roberta',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dropout(rate=0.1)(output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
# AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

# model.compile(
#     loss='sparse_categorical_crossentropy',
#     # optimizer=Adam(1e-5),  # 用足够小的学习率
#     optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
#         1000: 1,
#         2000: 0.1
#     }),
#     metrics=['accuracy'],
# )
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    preds = []
    golden = []
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        preds.extend(y_pred)
        golden.extend(y_true)
        right += (y_true == y_pred).sum()
    return right / total
    # return preds, golden


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(model_name)
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )

    model.load_weights(model_name)
    print(u'final test acc: %05f\n' % (evaluate(test_generator)))
    