#! -*- coding:utf-8 -*-

# 这里为了做rationale的实验
# x = Lambda(lambda x: K.reshape(x, (10, -1)))(x)
# if self.segment_vocab_size > 0:
#     s = Lambda(lambda x: K.reshape(x, (10, -1)))(s)
import numpy as np
import time
from bert4keras.backend import keras, K
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense, Dropout,Reshape,Input, Softmax
# tf.keras.layers.Reshape
from dataclasses import dataclass
from typing import List, Optional

import itertools
import json

set_gelu('tanh')  # 切换gelu版本

num_classes = 5
maxlen = 512
batch_size = 2
config_path = '$Anonymous_path$/bert_pretrained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = '$Anonymous_path$/bert_pretrained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = '$Anonymous_path$/bert_pretrained_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'

@dataclass(frozen=True)
class InputExample:
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]

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
    """加载数据
    单条格式：(文本, 标签id)
    """
    #rationale# 
    if "train" in filename:
        gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/train_prediction_mt5_rationale_generator_add_evidence.json"
    else:
        gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/prediction_add_evidence.mymodel_v3.json" # ours
        # gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v4_data/predict_to_human_label.json"
        # gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_rationale_generator_add_evidence.generate.json"
        # gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/prediction_add_evidence.mymodel_v6.json"
        # gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v4_data/predict.json" # w/ no-expl

        # gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_naive_generator_add_evidence.reasoning.json"
        # print(gf)
        # gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_naive_generator_add_evidence.reasoning.json" # wt-5
        gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_naive_generator.json" # w/o question context
        # exit(0)

        # gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_naive_generator_add_evidence.mt5.json"
    # q + a without evidence
    # gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_naive_generator.json"

    # q + a evidence
    # gf = "$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_naive_generator_add_evidence.reasoning.json"
    generated_expl = []
    with open(gf, encoding='utf-8') as f:
        r_data = json.load(f)
        for r in r_data: 
            pred_title = r['predict_explanation']
            pred_title = pred_title[pred_title.find("解析是：")+4:]
            # pred_title = pred_title[pred_title.find("。解析是：")+5:]
            # pred_title = pred_title[pred_title.find("解析:")+3:]
            # print(pred_title)
            generated_expl.append(pred_title)  
    D = []
    cnt = 0
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
            try:
                context_str = r['context'][0].replace(" ", "").replace("“", "\"").replace("”", "\"")
            except:
                context_str = "空"
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
                text = '{}？选项是:{}。参考:{}。'.format(r['question'], options_text, context_str) # cos-e
                # text = '{}？选项是:{}。'.format(r['question'], options_text) # cos-e
                # text = '参考文本为:{}。'.format(r['explanation'])
            for option in r['options']:
                text1 = '{}？'.format(r['question'])
                text2 = option
                label = trans(r['answer'])
            if type is None:
                print(type)
                D.append(InputExample(
                            question="",#r['question'],
                            # contexts=[r['context'][0],r['context'][1],r['context'][2],r['context'][3],r['context'][4]],  # this is not efficient but convenient
                            # contexts=[r['explanation'],r['explanation'],r['explanation'],r['explanation'],r['explanation']],
                            # contexts=[generated_expl[cnt], generated_expl[cnt+1], 
                            #             generated_expl[cnt+2], generated_expl[cnt+3], 
                            #             generated_expl[cnt+4]],
                            contexts=[generated_expl[cnt], generated_expl[cnt], 
                                        generated_expl[cnt], generated_expl[cnt], 
                                        generated_expl[cnt]],
                            # contexts=[None, None, None, None, None],
                            endings=[r['options'][0], r['options'][1], r['options'][2], r['options'][3], r['options'][4]],
                            
                            # endings=[r['options'][0]+" "+generated_expl[cnt], r['options'][1]+" "+generated_expl[cnt], r['options'][2]+" "+generated_expl[cnt], r['options'][3]+" "+generated_expl[cnt], r['options'][4]+" "+generated_expl[cnt]],
                            label=trans(r['answer']),
                        ))
            else:
                if type == "xe":
                    D.append(InputExample(
                            # question=r['question'],
                            question=r['question'],
                            # contexts=[r['context'][0],r['context'][1],r['context'][2],r['context'][3],r['context'][4]],  # this is not efficient but convenient
                            # contexts=[r['explanation'],r['explanation'],r['explanation'],r['explanation'],r['explanation']],
                            contexts=[generated_expl[cnt], generated_expl[cnt], 
                                        generated_expl[cnt], generated_expl[cnt], 
                                        generated_expl[cnt]],
                            # contexts=[generated_expl[cnt], generated_expl[cnt+1], 
                            #             generated_expl[cnt+2], generated_expl[cnt+3], 
                            #             generated_expl[cnt+4]],
                            endings=[r['options'][0], r['options'][1], r['options'][2], r['options'][3], r['options'][4]],
                            label=trans(r['answer']),
                        ))
                elif type == "e":
                    D.append(InputExample(
                            question=None,
                            # contexts=[r['context'][0],r['context'][1],r['context'][2],r['context'][3],r['context'][4]],  # this is not efficient but convenient
                            # contexts=[r['explanation'],r['explanation'],r['explanation'],r['explanation'],r['explanation']],
                            contexts=[generated_expl[cnt], generated_expl[cnt], 
                                        generated_expl[cnt], generated_expl[cnt], 
                                        generated_expl[cnt]],
                            # contexts=[generated_expl[cnt], generated_expl[cnt+1], 
                            #             generated_expl[cnt+2], generated_expl[cnt+3], 
                            #             generated_expl[cnt+4]],
                            endings=[None, None, None, None, None],
                            label=trans(r['answer']),
                        ))
                elif type == "x":
                    D.append(InputExample(
                            question=r['question'],
                            # contexts=[r['context'][0],r['context'][1],r['context'][2],r['context'][3],r['context'][4]],  # this is not efficient but convenient
                            contexts=[None,None,None,None,None],
                            # contexts=[generated_expl[cnt], generated_expl[cnt+1], 
                            #             generated_expl[cnt+2], generated_expl[cnt+3], 
                            #             generated_expl[cnt+4]],
                            endings=[r['options'][0], r['options'][1], r['options'][2], r['options'][3], r['options'][4]],
                            label=trans(r['answer']),
                        ))
            cnt += 1
    return D


# 加载数据集
# train_data = load_data('$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/train_context.json')
# valid_data = load_data('$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/test_context.json')
test_data = load_data('$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/test_context.json')
# train_data = load_data('$Anonymous_path$/Chinese-GPT/data/mydata/v1.9_data/train_context_v0_only_q.json')
# valid_data = load_data('$Anonymous_path$/Chinese-GPT/data/mydata/v1.9_data/test_context_v0_only_q.json')
# test_data = load_data('$Anonymous_path$/Chinese-GPT/data/mydata/v1.9_data/test_context_v0_only_q.json')
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, example in self.sample(random):
            token_ids_example = []
            segment_ids_example = []
            for item_ending, item_context in zip(example.endings, example.contexts):
                #e
                if example.question is None:
                    token_ids, segment_ids = tokenizer.encode(
                        item_context, None, maxlen=maxlen
                    )
                else:
                    if item_context is None:
                        token_ids, segment_ids = tokenizer.encode(
                            example.question + item_ending, None, maxlen=maxlen
                        )
                    else:
                        text1 = example.question + item_ending
                        text2 = item_context
                        # text2 = item_ending
                        # print(text1+text2)
                        token_ids, segment_ids = tokenizer.encode(
                            text1, text2, maxlen=maxlen
                        )
                segment_ids_example.append(segment_ids)
                token_ids_example.append(token_ids)
            batch_token_ids.append(token_ids_example)
            batch_segment_ids.append(segment_ids_example)
            batch_labels.append([example.label])
            
            if len(batch_token_ids) == self.batch_size or is_end:
                first_shape = len(batch_token_ids) 
                batch_token_ids = list(itertools.chain(*batch_token_ids))
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = list(itertools.chain(*batch_segment_ids))
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_token_ids = batch_token_ids.reshape((first_shape, num_classes, -1))
                batch_labels = sequence_padding(batch_labels)
                batch_segment_ids = batch_segment_ids.reshape((first_shape, num_classes, -1))
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型

bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='roberta',
    max_position=maxlen,
    return_keras_model=False,
)
# import ipdb; ipdb.set_trace()
# bert.model.input[0] = Lambda(lambda x: K.reshape(x, (batch_size*num_classes, -1)))(c[0])
# bert.model.input[1] = Lambda(lambda x: K.reshape(x, (batch_size*num_classes, -1)))(c[1])
# bert.model.input[0] = Input(Lambda(lambda x: K.reshape(x, (batch_size*num_classes, -1)))(bert.model.input[0]))
# bert.model.input[1] = Input(Lambda(lambda x: K.reshape(x, (batch_size*num_classes, -1)))(bert.model.input[1]))

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dropout(rate=0.1)(output)
output = Dense(
    units=1,
    # activation='softmax',
    kernel_initializer=bert.initializer,
)(output)

output = Lambda(lambda x: K.reshape(x, (-1, num_classes)))(output)
output = Softmax()(output)
# bert.model.input[0] = reshape(bert.model.input, (batch_size, num_classes, -1))
# import ipdb; ipdb.set_trace()
model = keras.models.Model(bert.model.input, output)

# model.summary()

# print(bert.model.input)


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
    optimizer=Adam(1e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)

# 转换数据集
# train_generator = data_generator(train_data, batch_size)
# valid_generator = data_generator(valid_data, batch_size)
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
    # print(preds)
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
            model.save_weights('best_model.weights.e')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )

def compute_sim(labels, xe, e, x, seed, print_results = False):
    labels = np.array(labels)
    xe = np.array(xe)
    e = np.array(e)
    x = np.array(x)
    xe_correct = np.array(1*(labels==xe))
    x_correct = np.array(1*(labels==x))
    e_correct = np.array(1*(labels==e))

    # baseline and leaking proxy variable
    baseline_correct = 1*(x_correct)
    leaking = 1*(e_correct)
    leaked = np.argwhere(leaking.tolist()).reshape(-1)
    
    # get subgroups
    nonleaked = np.setdiff1d(np.arange(len(e_correct)), leaked)
    xe_correct_leaked = xe_correct[leaked]
    e_correct_leaked = e_correct[leaked]
    x_correct_leaked = x_correct[leaked]
    xe_correct_nonleaked = xe_correct[nonleaked]
    e_correct_nonleaked = e_correct[nonleaked]
    x_correct_nonleaked = x_correct[nonleaked]
    num_leaked = len(leaked)
    num_non_leaked = len(xe) - num_leaked

    unweighted_mean = np.mean([np.mean(xe_correct[split]) - np.mean(baseline_correct[split]) for split in [leaked,nonleaked]])
    nonleaking_diff = np.mean(xe_correct_nonleaked) - np.mean(baseline_correct[nonleaked])
    leaking_diff = np.mean(xe_correct_leaked) - np.mean(baseline_correct[leaked])
    if print_results:
        print("\n------------------------")
        print("num (probably) leaked: %d" % num_leaked)
        print("y|x,e : %.4f    baseline : %.4f     y|x,e=null: %.4f" % (np.mean(xe_correct_leaked), np.mean(baseline_correct[leaked]), np.mean(x_correct_leaked)))
        print("diff: %.4f" % (leaking_diff))
        print()
        print("num (probably) nonleaked: %d" % num_non_leaked)
        print("y|x,e : %.4f    baseline : %.4f     y|x,e=null: %.4f" % (np.mean(xe_correct_nonleaked), np.mean(baseline_correct[nonleaked]), np.mean(x_correct_nonleaked)))
        print("diff: %.4f" % (nonleaking_diff))
        print()
        print("overall: ")
        print("y|x : %.4f      y|e : %.4f" % (np.mean(x_correct), np.mean(e_correct)))
        print("y|x,e: %.4f     baseline : %.4f" % (np.mean(xe_correct), np.mean(baseline_correct)))
        print("\nunweighted mean: %.2f" % (unweighted_mean*100))
        print("--------------------------")
    return unweighted_mean, leaking_diff, nonleaking_diff


if __name__ == '__main__':

    evaluator = Evaluator()

    # model.fit(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=10,
    #     callbacks=[evaluator]
    # )

    model.load_weights('best_model.weights.x.mrc.evidence')
    print(u'final test acc: %05f\n' % (evaluate(test_generator)))
    '''
    model.load_weights('best_model.weights.xe.mrc')
    valid_data = load_data('$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/test_context.json', type="xe")
    valid_generator = data_generator(valid_data, batch_size)
    preds_xe, golden = evaluate(valid_generator)

    model.load_weights('best_model.weights.x.mrc.one.seq')
    valid_data = load_data('$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/test_context.json', type="x")
    valid_generator = data_generator(valid_data, batch_size)
    preds_x, golden = evaluate(valid_generator)

    model.load_weights('best_model.weights.e')
    valid_data = load_data('$Anonymous_path$/Chinese-GPT/data/mydata/v2_data/test_context.json', type="e")
    valid_generator = data_generator(valid_data, batch_size)
    preds_e, golden = evaluate(valid_generator)

    compute_sim(golden, xe=preds_xe, e=preds_e, x=preds_x, seed=42, print_results = True)

    start = time.time()
    boot_times = 10000
    print(f"Starting bootstrap with {boot_times/1000:.0f}k samples...")
    leaking_diff_list = []
    nonleaking_diff_list = []
    overall_metric_list = []
    golden = np.array(golden)
    preds_e = np.array(preds_e)
    preds_x = np.array(preds_x)
    preds_xe = np.array(preds_xe)
    for b in range(boot_times):
        boot_idx = np.random.choice(np.arange(len(golden)), replace=True, size = len(golden))    
        ggolden = golden[boot_idx]
        ppreds_xe = preds_xe[boot_idx]
        ppreds_e = preds_e[boot_idx]
        ppreds_x = preds_x[boot_idx]

        mean, leaking_diff, nonleaking_diff = compute_sim(ggolden, xe=ppreds_xe, e=ppreds_e, x=ppreds_x, seed=42, print_results = False)
        overall_metric_list.append(mean)
        leaking_diff_list.append(leaking_diff)
        nonleaking_diff_list.append(nonleaking_diff)

    lb, ub = np.quantile(nonleaking_diff_list, (.025, .975))
    CI = (ub - lb) / 2
    print("\nnonleaking diff: %.2f (+/- %.2f)" % (np.mean(nonleaking_diff_list)*100, 100*CI))

    lb, ub = np.quantile(leaking_diff_list, (.025, .975))
    CI = (ub - lb) / 2
    print("\nleaking diff: %.2f (+/- %.2f)" % (np.mean(leaking_diff_list)*100, 100*CI))

    lb, ub = np.quantile(overall_metric_list, (.025, .975))
    CI = (ub - lb) / 2
    print("\nunweighted mean: %.2f (+/- %.2f)\n" % (np.mean(overall_metric_list)*100, 100*CI))

    print("time for bootstrap: %.1f minutes" % ((time.time() - start)/60))
    print("--------------------------\n")
    '''