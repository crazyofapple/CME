# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (c) 2020 Sawan Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modification history
# 2020 $Anonymous$ Modified to finetune and generate explanations for NLI
# Modification history
# 2020 $Anonymous$: Modified to finetune and generate explanations for Medical QA
# 2020 $Anonymous$: Add MRC model and KL divergence loss
# 2020 $Anonymous$: Add T5 generation model

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import json
import random
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from mymodel import SelfExplanationModel, SelfExplanationModelOnlyClassifier
from tqdm import tqdm, trange
from torch import nn

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, Adafactor,
                                  get_cosine_with_hard_restarts_schedule_with_warmup,
                                  BertConfig, BertLMHeadModel, BertTokenizer, XLMRobertaConfig, XLMRobertaTokenizer,
                                  AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMultipleChoice,
                                GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,BartForConditionalGeneration,XLMRobertaForCausalLM )
from transformers import BertForMultipleChoice, XLMRobertaTokenizer
from transformers import AutoTokenizer, T5Tokenizer
from model import (BertLMAddMrcHeadModel, AlbertLMAddMrcHeadModel, RobertaLMAddMrcHeadModel, 
                GPT2LMAddMrcHead, AlbertOneLMAddMrcHeadModel, NazaOneLMAddMrcHeadModel)
from lm_utils import (CLS_TOKEN, SEP_TOKEN, TSVAddMRCDataset, EOS_TOKEN, 
                    GPTTSVAddMRCDataset, ChineseMYDataset, computeBLEU, ChineseT5Dataset)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    # 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'bert': (BertConfig, BertLMAddMrcHeadModel, BertTokenizer),
    'albert': (AlbertConfig, AlbertLMAddMrcHeadModel, AlbertTokenizer),
    'albert-only': (AlbertConfig, AlbertOneLMAddMrcHeadModel, AlbertTokenizer),
    'roberta': (RobertaConfig, RobertaLMAddMrcHeadModel, RobertaTokenizer),
    'gpt': (GPT2Config, GPT2LMAddMrcHead, GPT2Tokenizer),
    'naza': (BertConfig, NazaOneLMAddMrcHeadModel, BertTokenizer),
    'xlm': (XLMRobertaConfig, XLMRobertaForCausalLM, XLMRobertaTokenizer),
    'SE': (RobertaConfig, SelfExplanationModel, BertTokenizer),
    'SE-only-classifier': (RobertaConfig, SelfExplanationModelOnlyClassifier, BertTokenizer)
}

cross_entropy_ignore_index = nn.CrossEntropyLoss().ignore_index # nn.CrossEntropyLoss().ignore_index if naza or gpt else 0 for T5

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, t5_tokenizer, bert_tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    # for n, p in model.named_parameters():
    #     print(n)
    optimizer_grouped_parameters_mt5 = [
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and "mt5" in n], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "mt5" in n], 'weight_decay': 0.0}
        ]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and ("mt5" not in n)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and ("mt5" not in n)], 'weight_decay': 0.0}
        ]
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    # warmup_proportion * t_total
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer_mt5 = Adafactor(optimizer_grouped_parameters_mt5, 
                    lr=0.001,  
                    scale_parameter=False,
                    relative_step=False,)
    # optimizer_mt5 = AdamW(optimizer_grouped_parameters_mt5, 
    #                 lr=2e-4,  
    #                 eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps*t_total, num_training_steps=t_total) 
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_accuracy = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            encoder_raw_inputs, decoder_raw_outputs, batch_mrc, batch_mask_mrc, batch_segment_mrc, labels_mrc = batch 
            encoder_raw_inputs = list(encoder_raw_inputs)
            decoder_raw_outputs = list(decoder_raw_outputs)

            t5_batch = t5_tokenizer.prepare_seq2seq_batch(src_texts=encoder_raw_inputs, tgt_texts=decoder_raw_outputs, return_tensors="pt", max_length=args.t5_max_length, max_target_length=args.t5_max_target_length)
            # e-snli
            # t5_batch = t5_tokenizer.prepare_seq2seq_batch(src_texts=encoder_raw_inputs, tgt_texts=decoder_raw_outputs, return_tensors="pt", max_length=175, max_target_length=120)
            input_ids=t5_batch.input_ids # mT5 encoder ids
            attention_mask=t5_batch.attention_mask # mT5 encoder attention mask
            labels = t5_batch.labels
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            

            batch_size = batch_mrc.size(0)
            max_seq_length = 0
            for i in range(batch_size):
                for j in range(batch_mrc.size(1)):
                    total_length = 0
                    for k in range(batch_mrc.size(2)):
                        if batch_mask_mrc[i, j, k] == 0:
                            break
                        total_length += 1   
                    max_seq_length = max(max_seq_length, total_length)
            batch_mrc = batch_mrc[:, :, :max_seq_length]
            batch_mask_mrc = batch_mask_mrc[:, :, :max_seq_length]
            batch_segment_mrc = batch_segment_mrc[:, :, :max_seq_length]
    
            labels_mrc = labels_mrc.to(args.device)
            batch_mrc = batch_mrc.to(args.device)
            batch_mask_mrc = batch_mask_mrc.to(args.device)
            batch_segment_mrc = batch_segment_mrc.to(args.device)
            
            padding_lengths = []
            for i in range(batch_size):
                flag = False
                for j in range(labels.size(1)):
                    if labels[i, j] == 0:
                        padding_lengths.append(j)
                        flag = True
                        break
                if not flag:
                    padding_lengths.append(-1)
                    
            # print(padding_lengths)
            for idx in range(len(padding_lengths)):
                if padding_lengths[idx] == -1:
                    continue
                labels[idx, padding_lengths[idx]:] = cross_entropy_ignore_index
            labels = labels.to(args.device)
            model.train()
            # 第1，2，3都是loss
            # print(batch_mask_mrc)
            # print(batch_segment_mrc)
            # print(batch_mrc)
            # print(labels_mrc)
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            attention_mask_mrc=batch_mask_mrc, 
                            token_type_ids_mrc=batch_segment_mrc,
                            input_ids_mrc=batch_mrc, 
                            labels=labels, 
                            labels_mrc=labels_mrc)
            
            if args.loss_type == "1":
                loss = outputs[0] + 0.1*outputs[1] + outputs[2] 
            elif args.loss_type == "2":
                loss = 0.1*outputs[0] + 0.1*outputs[1] + 0.8*outputs[2] 
            elif args.loss_type == "3":
                loss = outputs[0] + outputs[1] + outputs[2] 
            elif args.loss_type == "4":
                loss = outputs[2]
            # loss = 0.9*outputs[0] + 0.1*outputs[2] 
            # loss = 0.8*outputs[0] + 0.1*outputs[1] + 0.1*outputs[2] 
            # loss = outputs[0] + outputs[1] + outputs[2]
            # loss = outputs[1] + outputs[0]
            # loss = outputs[0] #+ outputs[1] 
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer_mt5.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, t5_tokenizer, bert_tokenizer)
                        for key, value in results.items():
                           tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss
                assert args.save_steps == args.logging_steps, "Save steps must equal to logging steps."
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    # Save model checkpoint
                    logger.info("Accuracy update: {} {}".format(results['mrc_accuracy'], best_accuracy))
                    if results['mrc_accuracy'] >= best_accuracy:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        best_accuracy = results['mrc_accuracy']
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step
 

def sample_sequence(model, length, context, mrc_context, batch_mask_mrc, batch_segment_mrc, device='cpu', sep_token_id=None,tokenizer=None):
    context = torch.tensor(context, dtype=torch.long, device=device)
    mrc_context = torch.tensor(mrc_context, dtype=torch.long, device=device)
    batch_mask_mrc = torch.tensor(batch_mask_mrc, dtype=torch.long, device=device)
    batch_segment_mrc = torch.tensor(batch_segment_mrc, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    past = None

    with torch.no_grad():
        for i in range(length):
            '''
            inputs = {'input_ids': generated}
            output, past = model(**inputs)
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits)
            generated = torch.cat((generated, next_token.view(1,1)), dim=1)
            if next_token.item() == eos_token_id:
                break
            context = next_token.view(1,1)
            '''
            # input_shape = generated.shape
            # attention_mask = generated.new_ones(input_shape)
            inputs = {'input_ids': generated, "attention_mask": None,\
                "attention_mask_mrc": batch_mask_mrc, "token_type_ids_mrc": batch_segment_mrc, \
                "input_ids_mrc": mrc_context, "return_dict": False}
            
            output= model(**inputs)[0] 
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits)
            generated = torch.cat((generated, next_token.view(1,1)), dim=1)
            if next_token.item() == sep_token_id:
                break
            context = next_token.view(1,1)
    return generated


def sample_sequence_v2(model, length, context, attention_mask, mrc_context, batch_mask_mrc, batch_segment_mrc, device='cpu', sep_token_id=None, tokenizer=None):
    context = torch.tensor(context, dtype=torch.long, device=device)
    mrc_context = torch.tensor(mrc_context, dtype=torch.long, device=device)
    batch_mask_mrc = torch.tensor(batch_mask_mrc, dtype=torch.long, device=device)
    batch_segment_mrc = torch.tensor(batch_segment_mrc, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    input_ids = context
    past = None
    '''
    temperature = 1.2 
  
    repetition_penalty=1.3 
    
    k= 30
    p= 0.95
    '''
    # output_sequences = model.mt5.generate(
    #     input_ids=, 
    #     attention_mask=attention_mask,
    #     max_length=120,
    #     # min_length=10,
    #     # temperature=1.1,
    #     # early_stopping=True,
    #     # top_k=30,
    #     # top_p=0.5,
    #     num_beams=5,
    #     # length_penalty=1.1,
    #     # repetition_penalty=1.3,
    #     # do_sample=True,
    #     # no_repeat_ngram_size=2,
    #     num_return_sequences=1,
    #     # input_ids_mrc=mrc_context,
    #     # attention_mask_mrc=batch_mask_mrc,
    #     # token_type_ids_mrc=batch_segment_mrc,
    # )
    output_sequences = model.mt5.generate(
                            input_ids=input_ids,
                            # attention_mask=attention_mask,
                            max_length=200,
                            num_beams=20,
                            # early_stopping=True,
                            num_return_sequences=1,
                            # repetition_penalty=1.5,
                            # no_repeat_ngram_size=2, 
                            # num_return_sequences=5, 
                            # early_stopping=True
                            # no_repeat_ngram_size=3,
                            # temperature=0.9,
            # top_k=top_k,
            # top_p=0.9,
            repetition_penalty=1.5,
            # do_sample=True,
    )
    # output_sequences = model.mt5.generate(
    #                             input_ids=input_ids,
    #                             attention_mask=attention_mask,
    #                             max_length=256,
    #                             do_sample=True,
    #                             top_k=40,
    #                             top_p=0.80,
    #                             num_return_sequences=3,
    #                             no_repeat_ngram_size=2,
    #                             early_stopping=True
    # )
    # print(output_sequences)
    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    return output_sequences


def generate(args, model, t5_tokenizer, bert_tokenizer, prefix=""):
    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    eval_output_dir = args.output_dir
    if args.model_type == "gpt":
        eval_dataset = GPTTSVAddMRCDataset(tokenizer, args, file_path=args.eval_data_file,
                                block_size=args.block_size, get_annotations=False)
    
    elif args.model_type[0:2] == "SE":
        eval_dataset = ChineseT5Dataset(t5_tokenizer, bert_tokenizer, args, file_path=args.eval_data_file,
                                    block_size=args.block_size, get_annotations=True, is_cosqa=args.is_cosqa, is_esnli=args.is_esnli)
    elif args.model_type == "naza":
        eval_dataset = ChineseMYDataset(tokenizer, args, file_path=args.eval_data_file,
                                block_size=args.block_size, get_annotations=False)
    else:
        eval_dataset = TSVAddMRCDataset(tokenizer, args, file_path=args.eval_data_file,
                                block_size=args.block_size, get_annotations=False)

    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    # Eval!
    logger.info("***** Running generation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))

    model.eval()
    with open(args.predict_json, "w") as wf:
        data = []
        sample_strs = []
        all_labels_strs = []
        for index, batch in enumerate(tqdm(eval_dataloader, desc="Generating")):
            example = {}
            encoder_raw_inputs, decoder_raw_outputs, batch_mrc, batch_mask_mrc, batch_segment_mrc, labels_mrc = batch 
            encoder_raw_inputs = list(encoder_raw_inputs)
            decoder_raw_outputs = list(decoder_raw_outputs)
            labels_strs = decoder_raw_outputs
            # print(encoder_raw_inputs)
            print(decoder_raw_outputs)
            t5_batch = t5_tokenizer.prepare_seq2seq_batch(src_texts=encoder_raw_inputs, tgt_texts=decoder_raw_outputs, return_tensors="pt", max_length=len(encoder_raw_inputs[0]), max_target_length=args.t5_max_target_length)
            input_ids=t5_batch.input_ids # mT5 encoder ids
            attention_mask=t5_batch.attention_mask # mT5 encoder attention mask
            labels = t5_batch.labels
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            batch_size = batch_mrc.size(0)
            max_seq_length = 0
            for i in range(batch_size):
                for j in range(batch_mrc.size(1)):
                    total_length = 0
                    for k in range(batch_mrc.size(2)):
                        if batch_mrc[i, j, k] == 0:
                            break
                        total_length += 1   
                    max_seq_length = max(max_seq_length, total_length)
            batch_mrc = batch_mrc[:, :, :max_seq_length]
            batch_mask_mrc = batch_mask_mrc[:, :, :max_seq_length]
            batch_segment_mrc = batch_segment_mrc[:, :, :max_seq_length]

            labels_mrc = labels_mrc.to(args.device)
            batch_mrc = batch_mrc.to(args.device)
            batch_mask_mrc = batch_mask_mrc.to(args.device)
            batch_segment_mrc = batch_segment_mrc.to(args.device)

            padding_lengths = []
            for i in range(batch_size):
                flag = False
                for j in range(labels.size(1)):
                    if labels[i, j] == 0:
                        padding_lengths.append(j)
                        flag = True
                        break
                if not flag:
                    padding_lengths.append(-1)
                    
            # print(padding_lengths)
            for idx in range(len(padding_lengths)):
                if padding_lengths[idx] == -1:
                    continue
                labels[idx, padding_lengths[idx]:] = cross_entropy_ignore_index
            labels = labels.to(args.device)

            input_ids = input_ids.squeeze() 
            batch_mrc = batch_mrc.squeeze()
            attention_mask = attention_mask.squeeze()
            batch_mask_mrc = batch_mask_mrc.squeeze()
            batch_segment_mrc = batch_segment_mrc.squeeze()
            example['id'] = index
            out = sample_sequence_v2(
                model=model,
                context=input_ids,
                attention_mask=attention_mask,
                mrc_context=batch_mrc,
                batch_mask_mrc=batch_mask_mrc,
                batch_segment_mrc=batch_segment_mrc,
                length=args.length,
                device=args.device,
                sep_token_id=1, # </s>对应于1
                tokenizer=t5_tokenizer
            )
            
            out = out[0, :].tolist()

            # text = t5_tokenizer.decode(out,skip_special_tokens=True, clean_up_tokenization_spaces=True)
            text = t5_tokenizer.decode(out)
            # prompt_text = tokenizer.decode(batch, clean_up_tokenization_spaces=True).replace(" ", "")
            # print(prompt_text)
            text = text.split("</s>")[0].strip()
            text = text.replace("<pad>", "")
            eval_dataset.add_explanation(index, text)
            if not (args.is_cosqa or args.is_esnli):
                text = text.replace(" ", "").replace(":", "：").replace("<unk>", "")
                sample_strs.append(" ".join(list(text)))
                labels_strs = [" ".join(list(labels_strs[0]))]
            else:
                s = "tells me that "
                # s = "because"
                text = text[text.find(s)+len(s):]
                labels_strs[0] = labels_strs[0][labels_strs[0].find(s)+len(s):]
                sample_strs.append(text)
                labels_strs = [labels_strs[0]]
            example['predict_explanation'] = text
            example['golden_explanation'] = labels_strs[0]
            data.append(example)
            # print(example)
            print("text: ", sample_strs[-1].replace("", ""))
            # json.dump(f, data, ensure_ascii=False, indent=2)
            
            
            all_labels_strs.extend(labels_strs)
            print("label_strs:" , all_labels_strs[-1].replace("", ""))
    
        bleu = computeBLEU(sample_strs, [[x] for x in all_labels_strs]) if len(sample_strs) > 0 else -1
        
        print("generate bleu: ", bleu)
        json.dump(data, wf, ensure_ascii=False, indent=2)
    

    #save
    # directory, filename = os.path.split(args.eval_data_file)
    # model_directory, model_name = os.path.split(os.path.normpath(args.output_dir))
    # output_name = os.path.join(directory, '{}_{}'.format(model_name, filename))
    # eval_dataset.save(output_name)

def accuracy(out, labels, predicted_list):
    outputs = np.argmax(out, axis=1)
    predicted_list.extend(list(outputs))
    return np.sum(outputs == labels)

def evaluate(args, model, t5_tokenizer, bert_tokenizer, prefix=""):
    eval_output_dir = args.output_dir
    if args.model_type == "gpt":
        eval_dataset = GPTTSVAddMRCDataset(tokenizer, args, file_path=args.eval_data_file,
                                block_size=args.block_size, get_annotations=True)
    elif args.model_type == "naza":
        eval_dataset = ChineseMYDataset(tokenizer, args, file_path=args.eval_data_file,
                                    block_size=args.block_size, get_annotations=True)
    elif args.model_type[0:2] == "SE":
        eval_dataset = ChineseT5Dataset(t5_tokenizer, bert_tokenizer, args, file_path=args.eval_data_file,
                                    block_size=args.block_size, get_annotations=True, is_cosqa=args.is_cosqa, is_esnli=args.is_esnli)
    else:
        eval_dataset = TSVAddMRCDataset(tokenizer, args, file_path=args.eval_data_file,
                                    block_size=args.block_size, get_annotations=False)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    eval_accuracy = 0.0
    
    eval_mse_loss = 0.0
    eval_mrc_loss = 0.0
    nb_eval_steps, nb_eval_examples = 0, 0
    model.eval()
    predicted_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        
        encoder_raw_inputs, decoder_raw_outputs, batch_mrc, batch_mask_mrc, batch_segment_mrc, labels_mrc = batch 
        encoder_raw_inputs = list(encoder_raw_inputs)
        decoder_raw_outputs = list(decoder_raw_outputs)
        t5_batch = t5_tokenizer.prepare_seq2seq_batch(src_texts=encoder_raw_inputs, tgt_texts=decoder_raw_outputs, return_tensors="pt", max_length=args.t5_max_length, max_target_length=args.t5_max_target_length)
        # e-snli
        # t5_batch = t5_tokenizer.prepare_seq2seq_batch(src_texts=encoder_raw_inputs, tgt_texts=decoder_raw_outputs, return_tensors="pt", max_length=175, max_target_length=120)
        input_ids=t5_batch.input_ids # mT5 encoder ids
        attention_mask=t5_batch.attention_mask # mT5 encoder attention mask
        labels = t5_batch.labels
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        batch_size = batch_mrc.size(0)
        max_seq_length = 0
        for i in range(batch_size):
            for j in range(batch_mrc.size(1)):
                total_length = 0
                for k in range(batch_mrc.size(2)):
                    if batch_mask_mrc[i, j, k] == 0:
                        break
                    total_length += 1   
                max_seq_length = max(max_seq_length, total_length)
        batch_mrc = batch_mrc[:, :, :max_seq_length]
        batch_mask_mrc = batch_mask_mrc[:, :, :max_seq_length]
        batch_segment_mrc = batch_segment_mrc[:, :, :max_seq_length]

        labels_mrc = labels_mrc.to(args.device)
        batch_mrc = batch_mrc.to(args.device)
        batch_mask_mrc = batch_mask_mrc.to(args.device)
        batch_segment_mrc = batch_segment_mrc.to(args.device)

    
        padding_lengths = []
        for i in range(batch_size):
            flag = False
            for j in range(labels.size(1)):
                if labels[i, j] == 0:
                    padding_lengths.append(j)
                    flag = True
                    break
            if not flag:
                padding_lengths.append(-1)
                
        # print(padding_lengths)
        for idx in range(len(padding_lengths)):
            if padding_lengths[idx] == -1:
                continue
            labels[idx, padding_lengths[idx]:] = cross_entropy_ignore_index
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            attention_mask_mrc=batch_mask_mrc, 
                            token_type_ids_mrc=batch_segment_mrc,
                            input_ids_mrc=batch_mrc, 
                            labels=labels, 
                            labels_mrc=labels_mrc)

            # mse mrc lm
            lm_loss = outputs[0]
            mse_loss = outputs[1]
            mrc_loss = outputs[2]
            if args.model_type == "gpt":
                logits = outputs[5]
            else:
                logits = outputs[4]
            logits = logits.detach().cpu().numpy()
            label_ids = labels_mrc.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids, predicted_list)
            if mse_loss is not None:
                eval_mse_loss += mse_loss.mean().item()
            eval_mrc_loss += mrc_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            if lm_loss is not None:
                eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
        nb_eval_examples += batch_size

    eval_loss = eval_loss / nb_eval_steps
    eval_mse_loss = eval_mse_loss / nb_eval_steps
    eval_mrc_loss = eval_mrc_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "mse_loss": eval_mse_loss,
        "mrc_loss": eval_mrc_loss,
        "mrc_accuracy": eval_accuracy,
        "perplexity": perplexity
    }
    predicted_list = [str(x) for x in predicted_list]
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("\n".join(predicted_list))

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument('--predict_json', type=str, default='', help="output json file path")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--t5_model_name_or_path", type=str,
                        help="The t5 model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--is_cosqa", action='store_true',
                        help="Whether to do e-snli")
    parser.add_argument("--is_esnli", action='store_true',
                        help="Whether to do commonsenseqa")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the eval data file")
    parser.add_argument("--do_generate", action='store_true',
                        help="Whether to generate text on the eval data file")
    parser.add_argument("--length", type=int, default=100,
                        help="Length for generation")
    parser.add_argument("--t5_max_length", type=int, default=512,
                        help="t5 encoder max length")
    parser.add_argument("--t5_max_target_length", type=int, default=120,
                        help="t5 decoder max length")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--data_type", default="tsv", type=str,
                        help="Dataset type")
    parser.add_argument('--num_choices', type=int, default=5,
                        help="number of choices")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--loss_type", default="1", type=str,
                        help="Loss type")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                            cache_dir=args.cache_dir if args.cache_dir else None, force_download=False)
    # t5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)
    # t5_tokenizer = T5Tokenizer.from_pretrained("/raid/$Anonymous_path$/mt5/my_mt5_base/sentencepiece_cn.model")
    # bert_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    # bert_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    bert_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, use_fast=False)
    model = model_class(config, args)
    # model = model_class.from_pretrained(args.output_dir)
    # model = model_class.from_pretrained("./mymodel_20201203-mymodel/checkpoint-2100")
    model.to(args.device)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        train_dataset = ChineseT5Dataset(t5_tokenizer, bert_tokenizer, args, file_path=args.train_data_file,
                                        block_size=args.block_size, get_annotations=True, is_cosqa=args.is_cosqa, is_esnli=args.is_esnli)

        if args.local_rank == 0:
            torch.distributed.barrier()
        print(type(model))
       
        global_step, tr_loss = train(args, train_dataset, model, t5_tokenizer, bert_tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        t5_tokenizer.save_pretrained(args.output_dir)
        bert_tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned

        model = model_class.from_pretrained(args.output_dir)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        model = model_class.from_pretrained(args.output_dir)
        # print(model)
        model.to(args.device)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        result = evaluate(args, model, t5_tokenizer, bert_tokenizer)

    #Generation
    if args.do_generate:
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        # t5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
        bert_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, use_fast=False)
        # bert_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        # t5_tokenizer = T5Tokenizer.from_pretrained("/raid/$Anonymous_path$/mt5/sentencepiece_cn.model")
        # bert_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        generate(args, model, t5_tokenizer, bert_tokenizer)


if __name__ == "__main__":
    main()
