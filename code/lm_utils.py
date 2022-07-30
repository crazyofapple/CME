import pandas as pd
import os
import pickle
import torch
import json
import re
from torch.utils.data import DataLoader, Dataset
from sacrebleu import corpus_bleu
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
PAD_TOKEN = '[PAD]'
EOS_TOKEN = '[EOS]'

def clear_explanation(x: str):
    c1 = re.compile("^[：|(1)|(2)|(3)|(1^)]+[\.]+")
    x = re.sub(c1, "", x)
    x = x.replace("（", "(").replace("）",")").replace("，", ",").replace(" ", "").replace("：", ":")
    return x 
    
def trans(ch):
    if ch == "1":
        return "A"
    elif ch == "2":
        return "B"
    elif ch == "3":
        return "C"
    elif ch == "4":
        return "D"
    elif ch == "5":
        return "E"
def ch_trans(ch):
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
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

# def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
#     """Uses sacrebleu's corpus_bleu implementation."""
#     return {"bleu": round(corpus_bleu(output_lns, refs_lns, **kwargs).score, 4)}

def computeBLEU(outputs, targets):
    # see https://github.com/mjpost/sacreBLEU
    targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return corpus_bleu(outputs, targets, lowercase=True).score

class TSVDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, get_annotations=False):
        self.print_count = 0
        self.sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        self.cls_token_id = tokenizer.convert_tokens_to_ids(CLS_TOKEN)
        self.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        cached_features_file, data = self.load_data(file_path, block_size)
        self.data = data

        if get_annotations: cached_features_file = cached_features_file + '_annotated'

        # if os.path.exists(cached_features_file):
        #     print ('Loading features from', cached_features_file)
        #     with open(cached_features_file, 'rb') as handle:
        #         self.examples = pickle.load(handle)
        #     return

        print ('Saving features from ', file_path, ' into ', cached_features_file) 

        def create_example(r):
            # text1 = '{} ? {}'.format( r['questions'], r['options']) # medical
            text1 = '{} {}'.format( r['questions'], r['options']) # cos-e
            tokenized_text1 = [self.cls_token_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1)) + [self.sep_token_id]
            prompt_length = len(tokenized_text1)
            tokenized_text, total_length = tokenized_text1, len(tokenized_text1)
            if get_annotations:
                text2 = r['explanations']
                tokenized_text2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text2))
                tokenized_text = tokenized_text1 + tokenized_text2
                tokenized_text = tokenized_text + [self.sep_token_id]
                total_length = len(tokenized_text)
                if len(tokenized_text) > block_size:
                    tokenized_text = tokenized_text[:block_size]
                if len(tokenized_text) < block_size:
                    tokenized_text = tokenized_text + [self.pad_token_id] * (block_size-len(tokenized_text))
            if self.print_count > 0:
                print(len(tokenized_text))
                print ('example: ', text1 + text2 if get_annotations else text1)
                self.print_count = self.print_count - 1
                print("total_length: ", total_length)
            return (tokenized_text, prompt_length, total_length)

        self.examples = self.data.apply(create_example, axis=1).to_list()
        
        print ('Saving ', len(self.examples), ' examples')
        with open(cached_features_file, 'wb') as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item][0]), self.examples[item][1], self.examples[item][2]

    def get_example_text(self, index):
        return self.data['prompt'][index]

    def add_explanation(self, index, explanation):
        explanation_name = 'Generated_Explanation'
        self.data.at[self.data.index[index], explanation_name] = explanation

    def load_data(self, file_path, block_size):
        assert os.path.isfile(file_path)
        # data = pd.read_csv(file_path, sep='\t', index_col='pairID')
        if ("csv" in file_path):
            data = pd.read_csv(file_path)
        elif "json" in file_path:
            data = pd.read_json(file_path)
        elif "tsv" in file_path:
            data = pd.read_csv(file_path, sep="\t")
        # print (data)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_{}_{}'.format(block_size, filename))
        return cached_features_file, data

    def save(self, filename):
        # self.data.to_csv(filename, sep='#')
        # self.data.to_json(filename, force_ascii=False)
        pass
        # with open(filename, "w") as f:
        #     data = self.data.to_dict('records')
        #     for i, exp in enumerate(data):
        #         exp['Generated_Explanation'] = exp['Generated_Explanation'].replace(" ", "")
        #         f.write(json.dumps(exp, ensure_ascii=False)+"\n")

class TSVAddMRCDataset(TSVDataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, get_annotations=False, is_training=False):
        self.print_count = 0
        self.token_between = ", "
        self.sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        self.cls_token_id = tokenizer.convert_tokens_to_ids(CLS_TOKEN)
        self.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        cached_features_file, data = self.load_data(file_path, block_size)
        self.data = data

        if get_annotations: cached_features_file = cached_features_file + '_annotated'

        if os.path.exists(cached_features_file):
            print ('Loading features from', cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
            return

        # print ('Saving features from ', file_path, ' into ', cached_features_file) 

        def create_example(r):
            # text1 = '{} ? {}'.format( r['questions'], r['options']) # medical
            # cos-e :
            options_text = self.token_between.join(r['options'].split("####"))
            text1 = '{} {}'.format( r['questions'], options_text) # cos-e
            tokenized_text1 = [self.cls_token_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1)) + [self.sep_token_id]
            prompt_length = len(tokenized_text1)
            tokenized_text, total_length = tokenized_text1, len(tokenized_text1)
            if get_annotations:
                text2 = r['explanations']
                tokenized_text2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text2))
                tokenized_text = tokenized_text1 + tokenized_text2
                tokenized_text = tokenized_text + [self.sep_token_id]
                
            if len(tokenized_text) > block_size:
                tokenized_text = tokenized_text[:block_size]
            total_length = len(tokenized_text)
            if len(tokenized_text) < block_size:
                tokenized_text = tokenized_text + [self.pad_token_id] * (block_size-len(tokenized_text))
            label_mrc = int(r['answers'][0]) - 1
            options_text = r['options'].split("####")

            input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc = self.convert_examples_to_features(r['questions'], \
                            options_text, tokenizer, block_size, r['options_input'] if 'options_input' in r else None) # cos-e
            # input_ids_mrc = self.convert_examples_to_features(r['questions'], r['options'].split("##"), tokenizer, block_size) # medical
            if self.print_count > 0:
                print(len(tokenized_text))
                print ('example: ', text1 + text2 if get_annotations else text1)
                self.print_count = self.print_count - 1
                print("total_length: ", total_length)
            # batch, bacth_mrc, prompt_lengths, total_lengths, labels_mrc
            return (tokenized_text, input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc, prompt_length, total_length, label_mrc)

        self.examples = self.data.apply(create_example, axis=1).to_list()
        # self.examples = self.examples[:100]

        # 先不用缓存的文件
        # print ('Saving ', len(self.examples), ' examples')
        # with open(cached_features_file, 'wb') as handle:
        #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def __getitem__(self, item):
        return torch.tensor(self.examples[item][0]), torch.tensor(self.examples[item][1]),  torch.tensor(self.examples[item][2]), \
            torch.tensor(self.examples[item][3]), self.examples[item][4], self.examples[item][5], self.examples[item][6]

    @staticmethod
    def convert_examples_to_features(start_ending, endings, tokenizer, max_seq_length, endings_text=None):
        """Loads a data file into a list of `InputBatch`s."""

        # CSQA is a multiple choice task. To perform this task using Bert,
        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given Swag example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # - [CLS] context [SEP] choice_5 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        # print(start_ending)
        start_ending_tokens = tokenizer.tokenize("Q: " + start_ending) # question tokens

        choices_ids = []
        choices_mask = []
        choices_segment_ids = []
        if endings_text is not None:
            endings_text = endings_text.split("######")
            assert len(endings_text) == len(endings)
        for ending_index, ending in enumerate(endings):
            if endings_text is None:
                ending_tokens = tokenizer.tokenize("A: " + ending)
                _truncate_seq_pair(start_ending_tokens, ending_tokens, max_seq_length - 3)
                tokens = ["[CLS]"] + start_ending_tokens + ["[SEP]"] + ending_tokens + ["[SEP]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                segment_ids = [0] * (len(start_ending_tokens) + 2) + [1] * (len(ending_tokens) + 1)

                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_ids.append(input_ids)
                choices_mask.append(input_mask)
                choices_segment_ids.append(segment_ids)
            else:
                
                ending_tokens = tokenizer.tokenize(endings_text[ending_index])
                ending_tokens = ending_tokens[:max_seq_length-2]
                tokens = ["[CLS]"] + ending_tokens + ["[SEP]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                segment_ids = [1] * (len(input_ids))

                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_ids.append(input_ids)
                choices_mask.append(input_mask)
                choices_segment_ids.append(segment_ids)
        return choices_ids, choices_mask, choices_segment_ids

class GPTTSVAddMRCDataset(TSVDataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, get_annotations=False, is_training=False):
        self.print_count = 0
        self.token_between = ", "
        self.sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        self.eos_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        cached_features_file, data = self.load_data(file_path, block_size)
        self.data = data

        if get_annotations: cached_features_file = cached_features_file + '_annotated'

        # if os.path.exists(cached_features_file):
        #     print ('Loading features from', cached_features_file)
        #     with open(cached_features_file, 'rb') as handle:
        #         self.examples = pickle.load(handle)
        #     return

        # print ('Saving features from ', file_path, ' into ', cached_features_file) 

        def create_example(r):
            # text1 = '{} ? {}'.format( r['questions'], r['options']) # medical
            # cos-e :
            options_text = self.token_between.join(r['options'].split("####"))
            text1 = '{} {}'.format(r['questions'], options_text) # cos-e
            tokenized_text1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1)) + [self.sep_token_id]
            prompt_length = len(tokenized_text1)
            tokenized_text, total_length = tokenized_text1, len(tokenized_text1)
            if get_annotations:
                text2 = r['explanations']
                tokenized_text2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text2))
                tokenized_text = tokenized_text1 + tokenized_text2
                tokenized_text = tokenized_text + [self.eos_token_id]
                total_length = len(tokenized_text)
                if len(tokenized_text) > block_size:
                    tokenized_text = tokenized_text[:block_size]
                if len(tokenized_text) < block_size:
                    tokenized_text = tokenized_text + [self.eos_token_id] * (block_size-len(tokenized_text))
            label_mrc = int(r['answers'][0]) - 1
            options_text = r['options'].split("####")

            input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc = self.convert_examples_to_features(r['questions'], \
                            options_text, tokenizer, block_size, r['options_input'] if 'options_input' in r else None) # cos-e
            # input_ids_mrc = self.convert_examples_to_features(r['questions'], r['options'].split("##"), tokenizer, block_size) # medical
            if self.print_count > 0:
                print(len(tokenized_text))
                print ('example: ', text1 + text2 if get_annotations else text1)
                self.print_count = self.print_count - 1
                print("total_length: ", total_length)
            # batch, bacth_mrc, prompt_lengths, total_lengths, labels_mrc
            # 只用前两个
            return (tokenized_text, input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc, prompt_length, total_length, label_mrc)

        self.examples = self.data.apply(create_example, axis=1).to_list()
        # self.examples = self.examples[:100]

        # 先不用缓存的文件
        # print ('Saving ', len(self.examples), ' examples')
        # with open(cached_features_file, 'wb') as handle:
        #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def __getitem__(self, item):
        return torch.tensor(self.examples[item][0]), torch.tensor(self.examples[item][1]),  torch.tensor(self.examples[item][2]), \
            torch.tensor(self.examples[item][3]), self.examples[item][4], self.examples[item][5], self.examples[item][6]

    def convert_examples_to_features(self, start_ending, endings, tokenizer, max_seq_length, endings_text=None):
        """Loads a data file into a list of `InputBatch`s."""

        # CSQA is a multiple choice task. To perform this task using Bert,
        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given Swag example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # - [CLS] context [SEP] choice_5 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        # print(start_ending)
        start_ending_tokens = tokenizer.tokenize("Q: " + start_ending) # question tokens

        choices_ids = []
        choices_mask = []
        choices_segment_ids = []
        if endings_text is not None:
            endings_text = endings_text.split("######")
            assert len(endings_text) == len(endings)
        for ending_index, ending in enumerate(endings):
            if endings_text is None:
                ending_tokens = tokenizer.tokenize("A: " + ending)
                _truncate_seq_pair(start_ending_tokens, ending_tokens, max_seq_length - 3)
                tokens = start_ending_tokens + ["[SEP]"] + ending_tokens + ["[EOS]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                segment_ids = [0] * (len(start_ending_tokens) + 2) + [1] * (len(ending_tokens) + 1)
                choices_mask.append(len(input_ids)-1)
                padding = self.eos_token_id * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_ids.append(input_ids)
                
                choices_segment_ids.append(segment_ids)
            else:
                ending_tokens = tokenizer.tokenize(endings_text[ending_index])
                ending_tokens = ending_tokens[:max_seq_length-2]
                tokens =  ending_tokens + ["[EOS]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                segment_ids = [1] * (len(input_ids))
                choices_mask.append(len(input_ids)-1)
                padding = [self.eos_token_id] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_ids.append(input_ids)
                
                choices_segment_ids.append(segment_ids)
        return choices_ids, choices_mask, choices_segment_ids

class ChineseMYDataset(TSVAddMRCDataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, get_annotations=False, is_training=False):
        self.print_count = 0
        self.token_between = "$"
        self.sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        self.cls_token_id = tokenizer.convert_tokens_to_ids(CLS_TOKEN)
        self.pad_token_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        cached_features_file, data = self.load_data(file_path, block_size)
        self.data = data

        if get_annotations: cached_features_file = cached_features_file + '_annotated'

        if os.path.exists(cached_features_file):
            print ('Loading features from', cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
            return

        print ('Saving features from ', file_path, ' into ', cached_features_file) 

        def create_example(r):
            r['question'] = r['question'].replace(" ", "").replace("“", "\"").replace("”", "\"")
            r['explanation'] = r['explanation'].replace(" ", "").replace("“", "\"").replace("”", "\"")
            # r['explanation'] = "该题应该选择选项{}。".format(r['answer']) #+ r['explanation']
            r['options'] = [x.strip().replace("“", "\"").replace("”", "\"") for x in r['options']]
        
            options_text = self.token_between.join(r['options'])
            options_text = ""
            abc = ["A", "B", "C", "D", "E"]
            for ch, option in zip(abc, r['options']):
                options_text += (ch + option)
            text1 = '{}?{}'.format(r['question'],options_text) # cos-e
            # print(text1)
            # tokenized_text = tokenizer(
            #         text1, 
            #         None,
            #         add_special_tokens=True,                    
            #         max_length=block_size,
            #         padding="max_length",
            #         truncation=True,
            #         return_overflowing_tokens=True,
            #     )['input_ids']
            
            tokenized_text = [self.cls_token_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1)) + [self.sep_token_id]
            prompt_length = 0
            # tokenized_text, total_length = tokenized_text1, len(tokenized_text1)
            if get_annotations:
                text2 = r['explanation']
                # text2 = tokenizer.tokenize(text2)
                # tokenized_text2 = tokenizer.convert_tokens_to_ids(text2)
                # tokenized_text = tokenized_text1 + tokenized_text2
                # print("##"*30)
                # print(text1 + text2)
                # print("##"*30)
                # tokenized_text = tokenized_text + [self.sep_token_id]
                tokenized_text = tokenizer(
                    text1,
                    text2, 
                    add_special_tokens=True,
                    max_length=block_size,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=True,
                )
                # dict_keys(['overflowing_tokens', 'num_truncated_tokens', 'input_ids', 'token_type_ids', 'attention_mask'])
                # print(tokenized_text['num_truncated_tokens'])
                tokenized_text= tokenized_text['input_ids']
            # if len(tokenized_text) > block_size:
            #     tokenized_text = tokenized_text[:block_size]
            # print(tokenized_text)
            total_length = 0
            for i in range(len(tokenized_text)):
                if tokenized_text[i] == 0:
                    break
                total_length += 1   
            for i in range(len(tokenized_text)):
                if tokenized_text[i] == self.sep_token_id:
                    prompt_length += 1 
                    break
                prompt_length += 1  
            # total_length = len(tokenized_text)
            # if len(tokenized_text) < block_size:
            #     tokenized_text = tokenized_text + [self.pad_token_id] * (block_size-len(tokenized_text))
            label_mrc = ch_trans(r['answer'])
            options_text = []
            for context_text, option_text in zip(r['context'], r['options']):
                options_text.append(context_text + "######" + option_text)
            
            input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc = self.convert_examples_to_features(r['question'], \
                            options_text, tokenizer, block_size, r['options_input'] if 'options_input' in r else None) # cos-e
            # options_text = []
            # for context_text, option_text in zip(r['context'], r['options']):
            #     options_text.append(r['explanation'] + "######" + option_text)
            # input_ids_explanation, inputs_mask_explanation, inputs_segment_explanation = self.convert_examples_to_features(r['question'], \
            #                 options_text, tokenizer, block_size, r['options_input'] if 'options_input' in r else None) 
            # input_ids_mrc = self.convert_examples_to_features(r['questions'], r['options'].split("##"), tokenizer, block_size) # medical
            if self.print_count > 0:
                print(len(tokenized_text))
                print ('example: ', text1 + text2 if get_annotations else text1)
                self.print_count = self.print_count - 1
                print("total_length: ", total_length)
            # batch, bacth_mrc, prompt_lengths, total_lengths, labels_mrc
            if get_annotations:
                return (tokenized_text, input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc, prompt_length, total_length, label_mrc)
            else:
                return (tokenized_text, input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc, prompt_length, total_length, r['explanation'])
        self.examples = self.data.apply(create_example, axis=1).to_list()
        print ('Saving ', len(self.examples), ' examples')
        with open(cached_features_file, 'wb') as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def convert_examples_to_features(self, start_ending, endings, tokenizer, max_seq_length, endings_text=None):
        """Loads a data file into a list of `InputBatch`s."""

        # CSQA is a multiple choice task. To perform this task using Bert,
        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given Swag example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # - [CLS] context [SEP] choice_5 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        # print(start_ending)
        

        choices_ids = []
        choices_mask = []
        choices_segment_ids = []
        if endings_text is not None:
            endings_text = endings_text.split("######")
            assert len(endings_text) == len(endings)
        for ending_index, ending in enumerate(endings):
            if endings_text is None:
                context, ending = ending.split("######")
                start_ending_tokens = tokenizer.tokenize(context)
                ending_tokens = tokenizer.tokenize(start_ending + "?" + ending)
                # _truncate_seq_pair(start_ending_tokens, ending_tokens, max_seq_length - 3)

                auto_inputs = tokenizer(
                    context,
                    start_ending + "?" + ending,
                    add_special_tokens=True,
                    max_length=max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=True,
                )
                # print(max_seq_length)
                # print(len(tokens))
                # exit(0)
                # print(auto_inputs.keys())
                
                input_ids = auto_inputs["input_ids"]
                input_mask = auto_inputs["attention_mask"]
                segment_ids = auto_inputs['attention_mask']
                # tokens = ["[CLS]"] + start_ending_tokens  + ["[SEP]"] + ending_tokens+ ["[SEP]"]
                # print(tokens)
                # exit(0)
                # input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # input_mask = [1] * len(input_ids)
                # segment_ids = [0] * (len(start_ending_tokens) + 2) + [1] * (len(ending_tokens) + 1)

                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_ids.append(input_ids)
                choices_mask.append(input_mask)
                choices_segment_ids.append(segment_ids)
            else:
                ending_tokens = tokenizer.tokenize(endings_text[ending_index])
                ending_tokens = ending_tokens[:max_seq_length-2]
                tokens =  ending_tokens + ["[EOS]"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                segment_ids = [1] * (len(input_ids))
                choices_mask.append(len(input_ids)-1)
                padding = [self.eos_token_id] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                choices_ids.append(input_ids)
                
                choices_segment_ids.append(segment_ids)
        return choices_ids, choices_mask, choices_segment_ids
    # def __getitem__(self, item):
    #     return torch.tensor(self.examples[item][0]), torch.tensor(self.examples[item][1]),  torch.tensor(self.examples[item][2]), \
    #         torch.tensor(self.examples[item][3]), torch.tensor(self.examples[item][4]), torch.tensor(self.examples[item][5]), self.examples[item][6]


class ChineseT5Dataset(TSVAddMRCDataset):
    def __init__(self, t5_tokenizer, bert_tokenizer, args, file_path='train', block_size=512, get_annotations=False, is_training=False, is_cosqa=False, is_esnli=False):
        self.print_count = 0
        self.token_between = "$"
        self.sep_token_id = bert_tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        self.cls_token_id = bert_tokenizer.convert_tokens_to_ids(CLS_TOKEN)
        self.pad_token_id = bert_tokenizer.convert_tokens_to_ids(PAD_TOKEN)
        self.t5_tokenizer = t5_tokenizer
        self.bert_tokenizer = bert_tokenizer
        # self.bert_tokenizer.do_lower_case = True
        cached_features_file, data = self.load_data(file_path, block_size)
        self.data = data

        if get_annotations: cached_features_file = cached_features_file + '_annotated_mymodel_roberta_lower_no_fast'
        # if get_annotations: cached_features_file = cached_features_file + '_annotated_mymodel_albert_no_fast'
        # cached_lm_175_train_data.csv_annotated_mymodel_albert_no_fast

        # 测试的时候注释掉，训练的时候去掉注释
        if os.path.exists(cached_features_file):
            print ('Loading features from', cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
                print("lens:{}".format(len(self.examples)))
            return
        print ('Saving features from ', file_path, ' into ', cached_features_file) 
        self.is_cosqa = is_cosqa
        self.is_esnli = is_esnli
        if not (self.is_cosqa or self.is_esnli):
            if "train" in file_path:
                gf ='/raid/ldf/Chinese-GPT/data/mydata/v1.9_data/add_train_context_v0_only_q.json'
            else:
                gf = '/raid/ldf/Chinese-GPT/data/mydata/v1.9_data/test_context_v0_only_q.json'
            gf_list = []
            with open(gf, "r") as gff:
                r_data = json.load(gff)
                for r in r_data: 
                    gf_list.append(r['context'][0])
            self.data['context_q'] = gf_list
        def create_example(r):
            # print(r['question'])
            if not (self.is_cosqa or self.is_esnli):
                label_mrc = ch_trans(r['answer'])
                r['question'] = r['question'].replace(" ", "").replace("“", "\"").replace("”", "\"")
                r['explanation'] = r['explanation'].replace(" ", "").replace("“", "\"").replace("”", "\"")
                r['options'] = [x.strip().replace("“", "\"").replace("”", "\"") for x in r['options']]
                context_str_q = r['context'][label_mrc].replace(" ", "").replace("“", "\"").replace("”", "\"").replace("\n", "").replace("\r","")
                # context_str_q = r['context_q'].replace(" ", "").replace("“", "\"").replace("”", "\"").replace("\n", "").replace("\r","")
            elif self.is_cosqa:
                # id,question,choice_0,choice_1,choice_2,label,human_exp
                if 'v1.1' in file_path:
                    r['options'] = [r['choice_0'], r['choice_1'], r['choice_2'], r['choice_3'],r['choice_4']]
                    r['context'] = ["", "", "", "", ""]
                else:
                    r['options'] = [r['choice_0'], r['choice_1'], r['choice_2']]
                    r['context'] = ["", "", ""]
                r['explanation'] = r['human_exp']
                
                
            elif self.is_esnli:
                r['question'] = "Premise: " + r['sentence1']
                r['options'] = ["Hypothesis: " + r['sentence2']]
                r['context'] = [""]
                if "Explanation_1" in r:
                    r['explanation'] = r['Explanation_1']
                label_map = {"neutral": 0,"entailment": 1, "contradiction":2}
                r['label'] = label_map[r['gold_label'].strip()]
            options_text = ""
            abc = ["A.", "B.", "C.", "D.", "E."]
            for ch, option in zip(abc, r['options']):
                options_text += (ch + option)
            
            if not (self.is_cosqa or self.is_esnli):
                label_mrc = ch_trans(r['answer'])
                # "question+options": "患者接受顺铂化疗时，药师应重点监测的不良反应是？选项是:A.心脏毒性B.肝脏毒性C.肾脏毒性D.肺纤维化E.过敏反应。参考文本为:（1）与顺铂合用应谨慎。博来霉素通过肾脏排泄占博来霉素总清除率的一半，而顺铂是有肾毒性药，可降低肾小球滤过率，影响博来霉素的清除。博来霉素的清除率下降会增强博来霉素肾毒性，后果严重。因此，两者合用时应经常监测肾功能，必要时减少博来霉素的剂量。。",
                # print(context_str_q)
                encoder_raw_inputs = '{}？选项:{}。参考:{}。'.format(r['question'], options_text.strip("。"), context_str_q.strip("。"))
                # encoder_raw_inputs = '{}？选项是{}。'.format(r['question'], options_text.strip("。"))
                # print(encoder_raw_inputs)
                # print(encoder_raw_inputs)
                if r['explanation'].strip() != "":
                   encoder_raw_inputs = "解释:" + encoder_raw_inputs
                if get_annotations:
                    if r['explanation'].strip() != "":
                        decoder_raw_inputs  = f"答案是{r['options'][label_mrc]}。解析是：" + r['explanation']
                    else:
                        decoder_raw_inputs  = f"答案是{r['options'][label_mrc]}。"
                    # decoder_raw_inputs  = f"解析是：" + r['explanation']
                    # print(decoder_raw_inputs)
                else:
                    decoder_raw_inputs = "" 
            elif self.is_cosqa:
                if 'v1.1' in file_path:
                    encoder_raw_inputs = f"{r['question']} The choices are {r['choice_0']}, {r['choice_1']}, {r['choice_2']}, {r['choice_3']} and {r['choice_4']}."
                else:
                    encoder_raw_inputs = f"{r['question']} The choices are {r['choice_0']}, {r['choice_1']} and {r['choice_2']}."
                label_mrc = r['label']
                # print(encoder_raw_inputs)
                # print(label_mrc)
                # print(f"The answer is {r['options'][label_mrc]} because " + r['explanation'])
                
                if get_annotations:
                    # decoder_raw_inputs  = f"The answer is {r['options'][label_mrc]} because " + r['explanation']
                    decoder_raw_inputs = "My commonsense tells me that " + r['explanation']
                else:
                    decoder_raw_inputs = ""
            elif self.is_esnli:
                encoder_raw_inputs = f"explain: nli Premise: {r['sentence1']} Hypothesis: {r['sentence2']}"
                label_mrc = r['label']
                if get_annotations: 
                    # label_map = {0: "neutral", 1: "entailment", 2: "contradiction"}
                    # label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
                    # decoder_raw_inputs = f"The answer is {r['gold_label'].strip()} because " + r['explanation']
                    decoder_raw_inputs = "My commonsense tells me that " + r['explanation']
                else:
                    decoder_raw_inputs = ""
            options_text = []
            
            for context_text, option_text in zip(r['context'], r['options']):
                options_text.append(context_text + "######" + option_text)
            
            input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc = self.convert_examples_to_features(r['question'], \
                            options_text, self.bert_tokenizer, block_size, r['options_input'] if 'options_input' in r else None) # cos-e
            
            
            return (encoder_raw_inputs, decoder_raw_inputs, input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc, label_mrc)
            # , axis=1
        if self.is_cosqa or self.is_esnli:
            self.examples = []
            for idx in range(len(self.data)):
                item = self.data.iloc[idx]
                self.examples += [create_example(item)]
        else:
            self.examples = self.data.apply(create_example, axis=1).to_list()
        
        print ('Saving ', len(self.examples), ' examples')
        with open(cached_features_file, 'wb') as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def convert_examples_to_features(self, start_ending, endings, tokenizer, max_seq_length, endings_text=None):
        choices_ids = []
        choices_mask = []
        choices_segment_ids = []
        assert endings_text is None
        for ending_index, ending in enumerate(endings):
            if endings_text is None:
                context, ending = ending.split("######")
                # _truncate_seq_pair(start_ending_tokens, ending_tokens, max_seq_length - 3)
                if context.strip() == "":
                    context = None
                # print(start_ending + ending)
                # exit(0)
                auto_inputs = tokenizer.encode_plus(
                    # start_ending + " " +  ending,
                    # context,
                    ("Q: " + start_ending).lower(), ("A: " + ending).lower(), 
                    add_special_tokens=True,
                    max_length=max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=True,
                )
                # print(auto_inputs.keys())
                input_ids = auto_inputs["input_ids"]
            
                input_mask = auto_inputs["attention_mask"]
                if "token_type_ids" not in auto_inputs:
                    segment_ids = [0] * len(input_ids)
                else:
                    segment_ids = auto_inputs['token_type_ids']
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                
                assert len(input_ids) == max_seq_length, print(len(input_ids))
                assert len(input_mask) == max_seq_length, print(len(input_mask))
                assert len(segment_ids) == max_seq_length, print(len(segment_ids))
                choices_ids.append(input_ids)
                choices_mask.append(input_mask)
                choices_segment_ids.append(segment_ids)
        return choices_ids, choices_mask, choices_segment_ids

    def __getitem__(self, item):
        return self.examples[item][0], self.examples[item][1], torch.tensor(self.examples[item][2]), \
            torch.tensor(self.examples[item][3]), torch.tensor(self.examples[item][4]), torch.tensor(self.examples[item][5])




if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
    from transformers import RobertaTokenizer, T5Tokenizer
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)
    # t5_tokenizer = T5Tokenizer.from_pretrained("/raid/ldf/mt5/my_mt5_base/sentencepiece_cn.model")
    # bert_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large", use_fast=False)
    bert_tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v2", use_fast=False)
    
    bert_tokenizer.do_lower_case = True
    # bert_tokenizer.decode(input_ids_mrc[0][0])
    train_dataset = ChineseT5Dataset(t5_tokenizer, bert_tokenizer, "", file_path="/raid/ldf/Chinese-GPT/LAS-NL-Explanations/sim_experiments/data/v1.1/train.csv",
                                        block_size=200, get_annotations=True, is_cosqa=True, is_esnli=False)
    
    train_sampler = RandomSampler(train_dataset)
    # t5_tokenizer.decode("</s>")
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=2)
    for batch in train_dataloader:
        encoder_raw_inputs, decoder_raw_inputs, input_ids_mrc, inputs_mask_mrc, inputs_segment_mrc, label_mrc = batch
        encoder_raw_inputs = list(encoder_raw_inputs)
        decoder_raw_inputs = list(decoder_raw_inputs)
        import ipdb; ipdb.set_trace()
        
        t5_batch = t5_tokenizer.prepare_seq2seq_batch(src_texts=encoder_raw_inputs, tgt_texts=decoder_raw_inputs, return_tensors="pt", max_length=512, max_target_length=512)
    
        print(t5_batch.input_ids)
        # print(t5_tokenizer.decode(t5_batch.input_ids.cpu().numpy()[0]))
        # print(t5_batch.attention_mask)
        print(t5_batch.labels)
        # print(t5_tokenizer.decode(t5_batch.labels.cpu().numpy()[0]))
        # print(encoder_raw_inputs)
        # print(decoder_raw_inputs)
        # print(input_ids_mrc[0][0].cpu().numpy())
        # print(bert_tokenizer.decode(input_ids_mrc[0][0].cpu().numpy()))
        # break
    # b = computeBLEU(["我 爱 你 我 爱 你 我 爱 你 我 爱 你 我 爱 你"], [["我 爱 你 我 爱 你 我 爱 你 我 爱 你 我 爱 你"]])
    # print(b)
