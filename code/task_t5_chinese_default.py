#! -*- coding: utf-8 -*-
# 微调多国语言版T5做Seq2Seq任务
# 介绍链接：kexue.fm/archives/7867
# 细节请看：https://github.com/bojone/t5_in_bert4keras
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l

from __future__ import print_function
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sacrebleu import corpus_bleu
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from myt5models import build_transformer_model
from bert4keras.tokenizers import SpTokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from keras.layers import Lambda, Dense, Dropout,Reshape,Input, Softmax, Layer
import time
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
# 基本参数
max_c_len = 1024
num_classes = 5
max_t_len = 150
batch_size = 8
hidden_size = 768
epochs = 50

# 模型路径
config_path = '/raid/$Anonymous$/mt5/mt5_base_config.json'
checkpoint_path = '/raid/$Anonymous$/mt5/mt5_base/model.ckpt-1000000'
# spm_path = '/raid/$Anonymous$/mt5/sentencepiece.model'
spm_path = '/raid/$Anonymous$/mt5/sentencepiece_cn.model'
keep_tokens_path = '/raid/$Anonymous$/mt5/sentencepiece_cn_keep_tokens.json'



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

def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    # data = pd.read_json(file_path)
    # with open(filename, encoding='utf-8') as f:
    #     for l in f:
    #         title, content = l.strip().split('\t')
    #         D.append((title, content))
    with open(filename, "r") as f:
        r_data = json.load(f)
        for r in r_data: 
            r['question'] = r['question'].replace(" ", "").replace("“", "\"").replace("”", "\"")
            r['explanation'] = r['explanation'].replace(" ", "").replace("“", "\"").replace("”", "\"")
            r['options'] = [x.strip().replace("“", "\"").replace("”", "\"") for x in r['options']]
            try:
                context_str = r['context'][0].replace(" ", "").replace("“", "\"").replace("”", "\"")
            except:
                context_str = "空"
            options_text = ",".join(r['options'])
            options_text = ""
            abc = ["A.", "B.", "C.", "D.", "E."]
            for ch, option in zip(abc, r['options']):
                options_text += (ch + option)
            text1 = '{}？选项是:{}。参考文本为:{}。'.format(r['question'],options_text, context_str) # cos-e
            # D.append(("此题的解析是"+r["explanation"], 
            #             text1, "此题的解析是", trans(r['answer'])))
            # D.append(("此题的答案是:"+r['options'][trans(r['answer'])].strip("。")+"。解析:"+r["explanation"], 
            #             text1, "此题的答案是:"+r['options'][trans(r['answer'])].strip("。")+"。解析:", trans(r['answer'])))
            D.append(r['options'][trans(r['answer'])].strip("。"))
            # print(D[-1])
    return D


# 加载数据集
# train_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/train_context.json')
# valid_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/test_context.json')
# test_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/test_context.json')
train_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v1.9_data/train_context_v0_only_q.json')
valid_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v1.9_data/test_context_v0_only_q.json')
test_data = load_data('/raid/$Anonymous$/Chinese-GPT/data/mydata/v1.9_data/test_context_v0_only_q.json')

# 加载分词器
# 
# tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
tokenizer = SpTokenizer(spm_path, token_end='</s>')
keep_tokens = json.load(open(keep_tokens_path))
# print(tokenizer.encode("[CLS]"))
# print(tokenizer.decode([x for x in range(500)]))
# exit(0)
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_c_token_ids, batch_t_token_ids, batch_labels = [], [], []

        for is_end, (title, content, _, label) in self.sample(random):
            
            c_token_ids, _ = tokenizer.encode(content, maxlen=max_c_len)
            t_token_ids, _ = tokenizer.encode(title, maxlen=max_t_len)
            batch_c_token_ids.append(c_token_ids)
            batch_t_token_ids.append([0] + t_token_ids)
            batch_labels.append([label])
            if len(batch_c_token_ids) == self.batch_size or is_end:
                batch_c_token_ids = sequence_padding(batch_c_token_ids)
                batch_t_token_ids = sequence_padding(batch_t_token_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_c_token_ids, batch_t_token_ids, batch_labels], None
                batch_c_token_ids, batch_t_token_ids, batch_labels = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def __init__(self, output_axis=None, weight1=0.5, weight2=0.5, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.output_axis = output_axis
        self.weight1 = weight1
        self.weight2 = weight2

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred, logits, labels = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = K.cast(mask[1], K.floatx())[:, :-1]  # 解码器自带mask
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        labels = Lambda(lambda x: K.reshape(x, (-1, )))(labels)
        loss = self.weight1 * loss + self.weight2 * K.sparse_categorical_crossentropy(labels, logits)
        return loss



t5 = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    keep_tokens=keep_tokens,
    model='t5.1.1',
    return_keras_model=False,
    name='T5',
)

class MRCLayer(Layer):
    def __init__(self, **kwargs):    
        super(MRCLayer, self).__init__(**kwargs)  
    
    def compute_mask(self, input, input_mask=None):   
        # do not pass the mask to the next layers   
        return None   
        
    def call(self, inputs, mask=None):
        d_output = Lambda(lambda x: x[:, 0], name='CLS-token')(inputs)
        d_output = Dropout(rate=0.1)(d_output)
        # d_output = Lambda(lambda x: tf.reshape(tf.reduce_mean(x, axis=1), (-1, hidden_size)), name='CLS-token')(inputs)
        d_output = Dense(
            units=num_classes,
            # activation='softmax',
            kernel_initializer=t5.initializer
        )(d_output)

        d_output = Lambda(lambda x: K.reshape(x, (-1, num_classes)))(d_output)
        # output = Lambda(lambda x: K.reshape(x, (-1, num_classes)))(output)
        d_output = Softmax()(d_output)
        return d_output


# (self.batch_size, self.sequence_length, self.hidden_size)
encoder = t5.encoder
decoder = t5.decoder
model = t5.model
mrc_layer = MRCLayer()
d_output = mrc_layer(t5.encoder.outputs[0])
# mrc_model = Model(t5.encoder.outputs[0], d_output)
output = CrossEntropy(1)([model.inputs[1], model.outputs[0], d_output, model.inputs[2]])

model = Model(model.inputs, output)
model.summary()

model.compile(
    optimizer=Adam(2e-4)
)


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        c_encoded = inputs[0]
        return decoder.predict([c_encoded, output_ids])[:, -1]

    def generate(self, text, answer_prompt, topk=1):
        c_token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        c_encoded = encoder.predict(np.array([c_token_ids]))[0]
        reshaped_c_encoded = tf.expand_dims(tf.convert_to_tensor(c_encoded), axis=0)
        pred_ans = mrc_layer.call(reshaped_c_encoded)
        pred_ans = K.eval(tf.argmax(pred_ans[0],axis=-1))
        # pred_ans = 0
        prompt_token_ids, _ = tokenizer.encode(answer_prompt)
        start_id=[0]+prompt_token_ids[:-1]
        # print(prompt_token_ids[:-1])
        self.first_output_ids = np.array([start_id])
        start_time = time.time()
        output_ids = self.beam_search([c_encoded], topk)  # 基于beam search
        print("decode for one example time {}.".format(time.time() - start_time))
        return tokenizer.decode([int(i) for i in output_ids]), pred_ans

# {'rouge-1': 0.21824932878093622, 'rouge-2': 0.12647243301008318, 'rouge-l': 0.28983179137237886, 'bleu': 7.779841388155039, 'accuracy': 0.21487603305785125}
# 注：T5有一个很让人不解的设置，它的<bos>标记id是0，即<bos>和<pad>其实都是0

autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=150)

def computeBLEU(outputs, targets):
    # see https://github.com/mjpost/sacreBLEU
    # targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return corpus_bleu(outputs, targets, lowercase=True).score


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            model.save_weights('./best_model.weights.wt5.multitask.20201126-17:30')  # 保存模型
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu, right = 0, 0, 0, 0, 0
        predict_data = []
        for title, content, answer_prompt, label in tqdm(data):
            total += 1
            ori_title = title
            ori_pred, pred_ans = autotitle.generate(content, answer_prompt, topk)
            print("pred_ans: {} vs label: {}".format(pred_ans, label))
            if int(pred_ans) == label:
                right += 1
            # title = ' '.join(ori_title.replace(answer_prompt, "")).lower()
            # pred_title = ' '.join(ori_pred.replace(answer_prompt, "")).lower()
            title = ' '.join(ori_title.replace(answer_prompt, "")).lower()
            pred_title = ' '.join(ori_pred.replace(answer_prompt, "")).lower()
            print(f"question+options: {content} #### predict_explanation: {ori_pred} ### golden_explanation: {ori_title}")
            print()
            predict_data.append({"question+options": content, "predict_explanation": ori_pred, "golden_explanation": ori_title})
            if pred_title.strip():
                
                scores = self.rouge.get_scores(hyps=pred_title, refs=title)

                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                # bleu += sentence_bleu(
                #     references=[title.split(' ')],
                #     hypothesis=pred_title.split(' '),
                #     smoothing_function=self.smooth
                # )
                # print(pred_title.split(' '))
                bleu += computeBLEU(outputs=pred_title, targets=[[title]])
                
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        acc = right / total
        metrics =  {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
            'accuracy': acc
        }
        # {'rouge-1': 0.5576561565574545, 'rouge-2': 0.4487316595082304, 'rouge-l': 0.585385000980327, 'bleu': 35.060763607145496}
        # with open("/raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_rationale_generator_add_evidence.generate.try2.json", "w") as f:
        #     json.dump(predict_data, f, ensure_ascii=False, indent=2)
        print(metrics)
        return metrics


if __name__ == '__main__':

    # evaluator = Evaluator()
    # train_generator = data_generator(train_data, batch_size)

    # model.fit(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=epochs,
    #     callbacks=[evaluator]
    # )
    # model.load_weights('./best_model.weights.wt5.multitask.20201126-17:30')
    # evaluator.evaluate(valid_data, topk=5)
    preds = [0, 4, 0, 4, 0, 2, 4, 4, 0, 3, 1, 1, 3, 4, 3, 0, 2, 2, 1, 0, 3, 3, 0, 4, 2, 4, 1, 3, 4, 3, 4, 0, 2, 0, 2, 1, 4, 1, 3, 3, 0, 2, 1, 0, 3, 0, 1, 2, 0, 0, 2, 3, 2, 4, 0, 0, 3, 1, 2, 4, 0, 0, 1, 4, 1, 1, 1, 0, 4, 0, 4, 1, 3, 4, 2, 2, 3, 3, 1, 4, 3, 1, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 1, 3, 2, 4, 2, 1, 3, 2, 1, 4, 2, 0, 3, 4, 0, 1, 0, 2, 0, 3, 0, 0, 1, 3, 0, 1, 3, 0, 3, 4, 0, 1, 3, 0, 4, 0, 1, 4, 3, 1, 3, 4, 2, 0, 1, 1, 4, 3, 4, 0, 3, 3, 3, 0, 0, 3, 4, 2, 4, 4, 3, 1, 0, 0, 2, 4, 2, 3, 0, 4, 1, 4, 0, 4, 1, 3, 4, 0, 3, 1, 1, 0, 4, 0, 0, 0, 0, 0, 2, 3, 2, 1, 3, 0, 4, 0, 2, 3, 4, 2, 1, 4, 4, 1, 3, 2, 1, 3, 4, 4, 0, 0, 1, 4, 3, 0, 1, 3, 1, 0, 2, 4, 1, 4, 3, 2, 1, 0, 3, 1, 0, 0, 1, 0, 1, 3, 1, 4, 4, 4, 1, 3, 1, 1, 3, 1, 3, 0, 1, 0]
    rouge = Rouge()
    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    right_num = 0
    bert_score = 0
    mymodel_preds_f = "/raid/$Anonymous$/Chinese-GPT/data/mydata/v4_data/predict.json"
    wt5_preds_f = "/raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_naive_generator_add_evidence.mt5.json"
    only_generate_preds_f = "/raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_naive_generator_add_evidence.reasoning.json"

    # /raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_naive_generator_add_evidence.mt5.json
    # /raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/prediction_add_evidence.mymodel_v3.json
    # /raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/prediction_mt5_rationale_generator_add_evidence.generate.json
    # 答案是米托蒽醌。解析是：
    gf = "/raid/$Anonymous$/Chinese-GPT/data/mydata/v4_data/predict_only_ans.json"
    # gf = "/raid/$Anonymous$/Chinese-GPT/data/mydata/v2_data/prediction_add_evidence.mymodel_v3.json"
    with open(gf, "r") as f:
        data = json.load(f)
        for ind, item in enumerate(data):
            total += 1

            pred_title = item['predict_explanation']
            title = item['golden_explanation'].replace(" ", "")
            pred_answer = pred_title[pred_title.find("答案是")+3:pred_title.find("解析是：")]
            pred_title = pred_title[pred_title.find("解析是：")+4:]
            # pred_title[pred_title.find("。解析是：")+5:]
            title = title[title.find("解析是：")+4:]
            print(title)
            print(pred_title)
            if pred_title.strip() == "":
                pred_title = " "
            print("&"*10)
            title = ' '.join(title).lower()

            pred_answer = pred_answer.strip("。")
            # print(test_data[ind])
            true_answer = test_data[ind].replace("＜", "<").replace("＞", ">")
            if pred_answer == true_answer:
                right_num += 1
            else:
                print("pred_answer: {} vs true answer: {}".format(pred_answer, true_answer))
            
            
            pred_title = ' '.join(pred_title).lower()
            bleu += computeBLEU(outputs=pred_title, targets=[[title]])
            # P_mul, R_mul, F_mul = score([pred_title], [[title]], lang="zh", rescale_with_baseline=True)
            # bert_score += F_mul.cpu().numpy()[0]
            scores = rouge.get_scores(hyps=pred_title, refs=title)

            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total
    # bert_score /= total
    accuracy = right_num / total
    metrics =  {
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l,
        'bleu': bleu,
        # "bert-score": bert_score,
        'accuracy': accuracy
    }
    print(metrics)
    
