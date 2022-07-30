import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import MT5ForConditionalGeneration
from transformers import AutoModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel,RobertaForSequenceClassification
from transformers.models.mt5.modeling_mt5 import MT5Config

from transformers.models.t5.modeling_t5 import T5Stack, T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import json
import copy

keep_tokens_path = '/raid/$Anonymous$/mt5/sentencepiece_cn_keep_tokens.json'
keep_tokens = json.load(open(keep_tokens_path))


def calc_mse_loss(mrc_outputs, explanation_outputs, mask=None):
    if mask is not None:
        # mask has False at padding_idx
        sel_mask = mask[:, :, None].expand_as(explanation_outputs).bool()
        s_logits_slct = torch.masked_select(explanation_outputs, sel_mask)
        t_logits_slct = torch.masked_select(mrc_outputs, sel_mask)
    else:
        t_logits_slct = mrc_outputs
        s_logits_slct = explanation_outputs
    return F.mse_loss(s_logits_slct, t_logits_slct)

def kd_mse_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the mse loss between logits_S and logits_T
    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    loss = F.mse_loss(beta_logits_S, beta_logits_T)
    return loss

def kd_ce_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the cross entropy between logits_S and logits_T
    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss

def calc_kl_div(mrc_outputs, explanation_outputs, temperature=1.0):
    loss_kl = F.kl_div(
            input=F.log_softmax(mrc_outputs / temperature, dim=-1),
            target=F.softmax(explanation_outputs / temperature, dim=-1),
            reduction="batchmean",
    ) * (temperature ** 2)
                
    return loss_kl


class MyMT5ForConditionalGeneration(MT5ForConditionalGeneration):
   

    model_type = "mt5"
    config_class = MT5Config
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]
    keys_to_never_save = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (sequence_output,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )




class SelfExplanationModel(RobertaPreTrainedModel):
    def __init__(self, config, args=None):
        super().__init__(config)
        # ./mymodel_20201203-mymodel/checkpoint-6300 hfl/chinese-roberta-wwm-ext-large google/mt5-small
        if args is None:
            self.num_choices = 5
        else:
            self.num_choices = args.num_choices
        # self.albert = AutoModel.from_pretrained("roberta-large",add_pooling_layer=False)
        if args is None:
            self.roberta = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        else:
            self.roberta = AutoModel.from_pretrained(args.model_name_or_path)
        # self.roberta = AutoModel.from_pretrained("albert-xxlarge-v2",add_pooling_layer=False)
        if args is None:
            self.mt5 = MyMT5ForConditionalGeneration.from_pretrained("/raid/$Anonymous$/mt5/my_mt5_base", from_tf=True)
        else:
            self.mt5 = MyMT5ForConditionalGeneration.from_pretrained(args.t5_model_name_or_path, from_tf=True)
        
        # self.mt5 = MyMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        # self.mt5 = MyMT5ForConditionalGeneration.from_pretrained("t5-base")
        
        # self.dropout = nn.Dropout(0.1)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        
        #e-snli
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(max(config.hidden_dropout_prob, 0.1))
        if (args is not None) and not (args.is_esnli):
            self.out_proj = nn.Linear(config.hidden_size, 1)
        else:
            self.out_proj = nn.Linear(config.hidden_size, self.num_choices)

        
        
        self.tg_classifier = nn.Linear(self.mt5.config.hidden_size, self.num_choices)
        # self.init_weights()
    def forward(
        self,
        input_ids=None, # mT5 encoder ids
        attention_mask=None, # mT5 encoder attention mask
        decoder_input_ids=None, # mT5 decoder ids
        decoder_attention_mask=None, # mT5 decoder attention mask
        labels=None, # mt5 labels
        input_ids_mrc=None, # roberta encoder ids
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        return_dict=None,
        **kwargs
    ):
        # assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        num_choices = self.num_choices
        t5_outputs = self.mt5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=False
        )
        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        # print("input_ids_mrc:", input_ids_mrc.shape)
        # alberta
        mrc_outputs = self.roberta(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
            return_dict=return_dict,
        )
        # mrc_outputs = self.roberta(
        #     input_ids_mrc,
        #     attention_mask=attention_mask_mrc,
        #     token_type_ids=token_type_ids_mrc,
        #     position_ids=position_ids_mrc,
        #     head_mask=head_mask_mrc,
        #     inputs_embeds=inputs_embeds_mrc,
        #     encoder_attention_mask=attention_mask_mrc,
        #     output_attentions=output_attentions_mrc,
        #     output_hidden_states=output_hidden_states_mrc,
        #     return_dict=return_dict,
        # )  
        # # outputs = t5_outputsx
        if labels_mrc is not None:
            
            explanation_classification_scores = self.tg_classifier(t5_outputs[1])
            explanation_classification_scores, _ = torch.max(explanation_classification_scores, dim=1)
            # print("explanation_classification_scores:", explanation_classification_scores.shape)
            # torch.max()
            
            #e-snli
            pooled_output_mrc = mrc_outputs[0]
            x = pooled_output_mrc[:, 0, :]  # take <s> token (equiv. to [CLS])
            # x = self.dropout(x)
            # x = self.dense(x)
            # x = torch.tanh(x)
            # x = self.dropout(x)
            logits_mrc = self.out_proj(x)
            
            # pooled_output_mrc = mrc_outputs[1]
            # pooled_output_mrc = self.dropout(pooled_output_mrc)
            # logits_mrc = self.classifier(pooled_output_mrc)
            reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            loss_fct_explanation = CrossEntropyLoss()
            cc_loss = loss_fct_explanation(explanation_classification_scores, labels_mrc)
            mse_loss = cc_loss + calc_kl_div(reshaped_logits_mrc, explanation_classification_scores, temperature=1)
            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            
            outputs = (t5_outputs[0],) + (mse_loss, ) + (loss_mrc, )  + (None, ) + (reshaped_logits_mrc, ) + outputs_mrc
            return outputs
       
        # 三个loss
        # 第一个是带解释的分类softmax损失+两个分布的KL loss（temperature=2.0)
        # 第二个是5个选项分数softmax的分类损失
        # 第三个是解释的生成损失
        # mse mrc lm
        #  output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #   return ((loss,) + output) if loss is not None else output
        return t5_outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)
    
    # def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
    #     return {"input_ids": input_ids, "attention_mask": None, \
    #                 "input_ids_mrc": model_kwargs["input_ids_mrc"], "attention_mask_mrc": model_kwargs["attention_mask_mrc"], "token_type_ids_mrc": model_kwargs["token_type_ids_mrc"]}
    # nohup 13 


class SelfExplanationModelOnlyClassifier(RobertaPreTrainedModel):
    def __init__(self, config, args=None):
        super().__init__(config)
        # ./mymodel_20201203-mymodel/checkpoint-6300 hfl/chinese-roberta-wwm-ext-large google/mt5-small

        self.num_choices = 5 #args.num_choices
        # self.albert = AutoModel.from_pretrained("roberta-large",add_pooling_layer=False)
        # if args is None:
        self.roberta = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        # else:
        #     self.roberta = AutoModel.from_pretrained(args.model_name_or_path)
        # self.roberta = AutoModel.from_pretrained("albert-xxlarge-v2",add_pooling_layer=False)
        # if args is None:
        #     self.mt5 = MyMT5ForConditionalGeneration.from_pretrained("/raid/$Anonymous$/mt5/my_mt5_base", from_tf=True)
        # else:
        #     self.mt5 = MyMT5ForConditionalGeneration.from_pretrained(args.t5_model_name_or_path)
        
        # self.mt5 = MyMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        # self.mt5 = MyMT5ForConditionalGeneration.from_pretrained("t5-base")
        
        # self.dropout = nn.Dropout(0.1)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        
        #e-snli
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(max(config.hidden_dropout_prob, 0.1))
        self.out_proj = nn.Linear(config.hidden_size, 1)
        
        
        # self.tg_classifier = nn.Linear(self.mt5.config.hidden_size, self.num_choices)
        # self.init_weights()
    def forward(
        self,
        input_ids=None, # mT5 encoder ids
        attention_mask=None, # mT5 encoder attention mask
        decoder_input_ids=None, # mT5 decoder ids
        decoder_attention_mask=None, # mT5 decoder attention mask
        labels=None, # mt5 labels
        input_ids_mrc=None, # roberta encoder ids
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        return_dict=None,
        **kwargs
    ):
        # assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        num_choices = self.num_choices
        # t5_outputs = self.mt5(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     decoder_input_ids=decoder_input_ids,
        #     decoder_attention_mask=decoder_attention_mask,
        #     labels=labels,
        #     return_dict=False
        # )
        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        # print("input_ids_mrc:", input_ids_mrc.shape)
        # alberta
        mrc_outputs = self.roberta(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
            return_dict=return_dict,
        )
        # mrc_outputs = self.roberta(
        #     input_ids_mrc,
        #     attention_mask=attention_mask_mrc,
        #     token_type_ids=token_type_ids_mrc,
        #     position_ids=position_ids_mrc,
        #     head_mask=head_mask_mrc,
        #     inputs_embeds=inputs_embeds_mrc,
        #     encoder_attention_mask=attention_mask_mrc,
        #     output_attentions=output_attentions_mrc,
        #     output_hidden_states=output_hidden_states_mrc,
        #     return_dict=return_dict,
        # )  
        # # outputs = t5_outputsx
        if labels_mrc is not None:
            # x, _ = torch.max(t5_outputs[1], dim=1)
            # explanation_classification_scores = self.tg_classifier(x)
            # torch.max()
            
            #e-snli
            pooled_output_mrc = mrc_outputs[0]
            x = pooled_output_mrc[:, 0, :]  # take <s> token (equiv. to [CLS])
            # x = self.dropout(x)
            # x = self.dense(x)
            # x = torch.tanh(x)
            # x = self.dropout(x)
            logits_mrc = self.out_proj(x)
            
            # pooled_output_mrc = mrc_outputs[1]
            # pooled_output_mrc = self.dropout(pooled_output_mrc)
            # logits_mrc = self.classifier(pooled_output_mrc)
            reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            # loss_fct_explanation = CrossEntropyLoss()
            # cc_loss = loss_fct_explanation(explanation_classification_scores, labels_mrc)
            # mse_loss = cc_loss + calc_kl_div(reshaped_logits_mrc, explanation_classification_scores, temperature=1)
            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            
            outputs = (None,) + (None, ) + (loss_mrc, )  + (None, ) + (reshaped_logits_mrc, ) + outputs_mrc
            return outputs
       
        # 三个loss
        # 第一个是带解释的分类softmax损失+两个分布的KL loss（temperature=2.0)
        # 第二个是5个选项分数softmax的分类损失
        # 第三个是解释的生成损失
        # mse mrc lm
        #  output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #   return ((loss,) + output) if loss is not None else output
        return None # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)
    
    # def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
    #     return {"input_ids": input_ids, "attention_mask": None, \
    #                 "input_ids_mrc": model_kwargs["input_ids_mrc"], "attention_mask_mrc": model_kwargs["attention_mask_mrc"], "token_type_ids_mrc": model_kwargs["token_type_ids_mrc"]}
    # nohup 13 
