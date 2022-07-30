
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from copy import deepcopy
import math
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  BertConfig, BertLMHeadModel, BertTokenizer, GPT2LMHeadModel, 
                                  GPT2Model, BertForMaskedLM, BertForMultipleChoice, XLMRobertaForMultipleChoice,
                                  BartForConditionalGeneration, BartModel, GPT2PreTrainedModel,
                                  GPT2DoubleHeadsModel, AlbertPreTrainedModel, BertForPreTraining, BertPreTrainedModel, CTRLLMHeadModel)
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertModel
from transformers.models.albert.modeling_albert import AlbertModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_utils import SequenceSummary
from transformers.modeling_outputs import CausalLMOutputWithPast

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

def calc_kl_div(mrc_outputs, explanation_outputs, temperature=1.0):
    loss_kl = F.kl_div(
            input=F.log_softmax(mrc_outputs / temperature, dim=-1),
            target=F.softmax(explanation_outputs / temperature, dim=-1),
            reduction="batchmean",
    ) * (temperature ** 2)
                
    return loss_kl

class AttentionMerge(nn.Module):
    """
    H (B, L, hidden_size) => h (B, hidden_size)
    """
    def __init__(self, input_size, attention_size, dropout_prob):
        super(AttentionMerge, self).__init__()
        self.attention_size = attention_size
        self.hidden_layer = nn.Linear(input_size, self.attention_size)
        self.query_ = nn.Parameter(torch.Tensor(self.attention_size, 1))
        self.dropout = nn.Dropout(dropout_prob)

        self.query_.data.normal_(mean=0.0, std=0.02)

    def forward(self, values, mask=None):
        """
        (b, l, h) -> (b, h)
        """
        if mask is None:
            mask = torch.zeros_like(values)
            # mask = mask.data.normal_(mean=0.0, std=0.02)
        else:
            mask = (1 - mask.unsqueeze(-1).type(torch.float)) * -1000.

        keys = self.hidden_layer(values)
        keys = torch.tanh(keys)
        query_var = torch.var(self.query_)
        # (b, l, h) + (h, 1) -> (b, l, 1)
        attention_probs = keys @ self.query_ / math.sqrt(self.attention_size * query_var)
        # attention_probs = keys @ self.query_ / math.sqrt(self.attention_size)

        attention_probs = F.softmax(attention_probs * mask, dim=1)
        attention_probs = self.dropout(attention_probs)

        context = torch.sum(attention_probs + values, dim=1)
        return context


class BertLMAddMrcHeadModel(BertModel):
    '''
    Bert Model with a language modeling head on top for CLM fine-tuning. 
    CLM stands for Causal Language Modeling in which a given word 
    is trained based only on the previous words and not using the masking technique.
    This model is a PyTorch torch.nn.Module sub-class. 
    Use it as a regular PyTorch Module and 
    refer to the PyTorch documentation for all matter related to general usage and behavio
    '''
    def __init__(self, config):
        # config.update({'is_decoder': True})
        super().__init__(config)
        # assert config.is_decoder, "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`."
        self.num_choices = 5
        config.update({'is_decoder': True})
        self.bert = BertModel(config)
        config_mrc = deepcopy(config)
        config_mrc.update({'is_decoder': False})  
        self.mrc_bert = BertModel(config_mrc)
        self.cls = BertOnlyMLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.tg_classifier = nn.Linear(config.hidden_size, self.num_choices)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        explanation_classification_scores = self.tg_classifier(outputs[1])
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        mrc_outputs = self.mrc_bert(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            encoder_hidden_states=encoder_hidden_states_mrc,
            encoder_attention_mask=encoder_attention_mask_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
        )
        if labels_mrc is not None:
            pooled_output_mrc = mrc_outputs[1]
            pooled_output_mrc = self.dropout(pooled_output_mrc)
            logits_mrc = self.classifier(pooled_output_mrc)
            reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            
            mse_loss = calc_kl_div(reshaped_logits_mrc, explanation_classification_scores)

            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            
            outputs = outputs + outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs

            outputs = (mse_loss, ) + outputs
        # mse mrc lm
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)


class GPT2LMAddMrcHead(GPT2DoubleHeadsModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_choices = 5
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        config.summary_type = "cls_index"
        config.num_labels = 1
        self.multiple_choice_head = SequenceSummary(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        
        # explanation_classification_scores = self.tg_classifier(outputs[1])
        # outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,

        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        # explanation_classification_scores = self.tg_classifier(outputs[1])
        # outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1,) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        mrc_outputs = self.transformer(
            input_ids_mrc,
            past=None, 
            attention_mask=None,
            token_type_ids=None,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
        )
        hidden_states = mrc_outputs[0]
        if labels_mrc is not None:
            # mc_token_ids
            mc_logits = self.multiple_choice_head(hidden_states, attention_mask_mrc).squeeze(-1)
            # print("#######: ", mc_logits.shape)
            reshaped_logits_mrc = mc_logits.view(-1, num_choices)
            # mse_loss = calc_kl_div(reshaped_logits_mrc, explanation_classification_scores)
            # print("#######: ", reshaped_logits_mrc.shape)
            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            # print("len: ", len(outputs))
            outputs = outputs + outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs

            outputs = (loss_mrc, ) + outputs
        # mse mrc lm
        # print(outputs[5].shape)
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)


class AlbertLMAddMrcHeadModel(AlbertPreTrainedModel):
    '''
    Bert Model with a language modeling head on top for CLM fine-tuning. 
    CLM stands for Causal Language Modeling in which a given word 
    is trained based only on the previous words and not using the masking technique.
    This model is a PyTorch torch.nn.Module sub-class. 
    Use it as a regular PyTorch Module and 
    refer to the PyTorch documentation for all matter related to general usage and behavio
    '''
    def __init__(self, config):
        # config.update({'is_decoder': True})
        super().__init__(config)
        # assert config.is_decoder, "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`."
        self.num_choices = 5
        # config_lm.update({'is_decoder': True})
        # self.bert = BertModel(config_lm)
        # self.config_lm = config_lm
        self.albert = AlbertModel(config)
        # self.cls = BertOnlyMLMHead(config_lm)
        # config_mrc.hidden_dropout_prob = 0.1
        self.att_merge = AttentionMerge(config.hidden_size, attention_size=1024, dropout_prob=0.1)
        # self.dropout = nn.Dropout(0.1)
        self.scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1)
        )
        # self.classifier = nn.Linear(config.hidden_size, 1)
        # self.tg_classifier = nn.Linear(config_lm.hidden_size, self.num_choices)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=encoder_attention_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        # )

        # sequence_output = outputs[0]
        # prediction_scores = self.cls(sequence_output)
        # explanation_classification_scores = self.tg_classifier(outputs[1])
        # outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        
        # if labels is not None:
        #     # we are doing next-token prediction; shift prediction scores and input ids by one
        #     prediction_scores = prediction_scores[:, :-1, :].contiguous()
        #     labels = labels[:, 1:].contiguous()
        #     loss_fct = CrossEntropyLoss()
        #     ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config_lm.vocab_size), labels.view(-1))
        #     outputs = (ltr_lm_loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        # self,
        # input_ids=None,
        # attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # output_attentions=None,
        # output_hidden_states=None,
        mrc_outputs = self.albert(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            # position_ids=position_ids_mrc,
            # head_mask=head_mask_mrc,
            # inputs_embeds=inputs_embeds_mrc,
            # encoder_hidden_states=encoder_hidden_states_mrc,
            # encoder_attention_mask=encoder_attention_mask_mrc,
            # output_attentions=output_attentions_mrc,
            # output_hidden_states=output_hidden_states_mrc,
        )
        
        if labels_mrc is not None:
            # pooled_output_mrc = mrc_outputs[1]
            # pooled_output_mrc = self.dropout(pooled_output_mrc)
            # logits_mrc = self.classifier(pooled_output_mrc)
            h12 = self.att_merge(mrc_outputs[0], attention_mask_mrc)
            logits_mrc = self.scorer(h12)
            reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            
            # mse_loss = calc_kl_div(reshaped_logits_mrc, explanation_classification_scores)

            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            
            outputs = outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs
            outputs =  (loss_mrc,)  + outputs
            outputs = (loss_mrc, ) + outputs
            outputs = (loss_mrc, ) + outputs
        # mse mrc lm
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)


class RobertaLMAddMrcHeadModel(RobertaModel):
    '''
    Bert Model with a language modeling head on top for CLM fine-tuning. 
    CLM stands for Causal Language Modeling in which a given word 
    is trained based only on the previous words and not using the masking technique.
    This model is a PyTorch torch.nn.Module sub-class. 
    Use it as a regular PyTorch Module and 
    refer to the PyTorch documentation for all matter related to general usage and behavio
    '''
    def __init__(self, config):
        # config.update({'is_decoder': True})
        super().__init__(config)
        # assert config.is_decoder, "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`."
        self.num_choices = 5
        config.update({'is_decoder': True})
        self.bert = BertModel(config)
        config_mrc = deepcopy(config)
        config_mrc.update({'is_decoder': False})  
        self.mrc_bert = RobertaModel(config_mrc)
        self.cls = BertOnlyMLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.tg_classifier = nn.Linear(config.hidden_size, self.num_choices)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        explanation_classification_scores = self.tg_classifier(outputs[1])
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        # self,
        # input_ids=None,
        # attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # output_attentions=None,
        # output_hidden_states=None,
        mrc_outputs = self.mrc_bert(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            # encoder_hidden_states=encoder_hidden_states_mrc,
            # encoder_attention_mask=encoder_attention_mask_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
        )
        if labels_mrc is not None:
            pooled_output_mrc = mrc_outputs[1]
            pooled_output_mrc = self.dropout(pooled_output_mrc)
            logits_mrc = self.classifier(pooled_output_mrc)
            reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            
            mse_loss = calc_mse_loss(reshaped_logits_mrc, explanation_classification_scores)

            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            
            outputs = outputs + outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs

            outputs = (mse_loss, ) + outputs
        # mse mrc lm
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)


class BartLMAddMrcHeadModel(BartModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_choices = 5
        self.model = BartModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.tg_classifier = nn.Linear(config.hidden_size, self.num_choices)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        explanation_classification_scores = self.tg_classifier(outputs[1])
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        mrc_outputs = self.mrc_bert(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            encoder_hidden_states=encoder_hidden_states_mrc,
            encoder_attention_mask=encoder_attention_mask_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
        )
        if labels_mrc is not None:
            pooled_output_mrc = mrc_outputs[1]
            pooled_output_mrc = self.dropout(pooled_output_mrc)
            logits_mrc = self.classifier(pooled_output_mrc)
            reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            
            mse_loss = calc_kl_div(reshaped_logits_mrc, explanation_classification_scores)

            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            
            outputs = outputs + outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs

            outputs = (mse_loss, ) + outputs
        # mse mrc lm
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)


class AlbertOneLMAddMrcHeadModel(AlbertPreTrainedModel):
    def __init__(self, config):
        # config.update({'is_decoder': True})
        super().__init__(config)
        # assert config.is_decoder, "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`."
        self.num_choices = 5
        config.update({'is_decoder': True})
        self.albert = AlbertModel(config)
        self.cls = BertOnlyMLMHead(config)
        # self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        config.summary_type = "cls_index"
        config.summary_proj_to_labels = True
        config.summary_use_proj = True
        config.summary_first_dropout = 0.15
        config.num_labels = 1
        self.config = config
        self.multiple_choice_head = SequenceSummary(config)
        # self.att_merge = AttentionMerge(config.hidden_size, attention_size=1024, dropout_prob=0.1)
        # self.dropout = nn.Dropout(0.1)
        # self.scorer = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hidden_size, 1)
        # )
        config.num_labels = 5
        self.explanation_head = SequenceSummary(config)
        # self.att_merge_2 = AttentionMerge(config.hidden_size, attention_size=1024, dropout_prob=0.1)
        # self.scorer_2 = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hidden_size, self.num_choices)
        # )
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pre = torch.sum(attention_mask, axis=-1) 
        attention_length = pre - torch.ones_like(pre)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        # explanation_classification_scores = self.tg_classifier(outputs[1])

        explanation_classification_scores = self.explanation_head(sequence_output, attention_length).squeeze(-1)
        # h_exp = self.att_merge_2(outputs[0], attention_mask)
        # explanation_classification_scores = self.scorer_2(h_exp)
            
        explanation_classification_scores = explanation_classification_scores.view(-1, num_choices)
        # print("explanation_classification_scores: ", explanation_classification_scores.shape)
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        # self,
        # input_ids=None,
        # attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # output_attentions=None,
        # output_hidden_states=None,
        mrc_outputs = self.albert(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            # encoder_hidden_states=encoder_hidden_states_mrc,
            # encoder_attention_mask=encoder_attention_mask_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
        )
        hidden_states = mrc_outputs[0]
        pre = torch.sum(attention_mask_mrc, axis=-1) 
        attention_mrc_length = pre - torch.ones_like(pre)
        # print(attention_mrc_length)
        if labels_mrc is not None:
            # pooled_output_mrc = mrc_outputs[1]
            # pooled_output_mrc = self.dropout(pooled_output_mrc)
            # logits_mrc = self.classifier(pooled_output_mrc)
            # reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            # print(hidden_states.shape)
            mc_logits = self.multiple_choice_head(hidden_states, attention_mrc_length).squeeze(-1)

            # h12 = self.att_merge(mrc_outputs[0], attention_mask_mrc)
            # mc_logits = self.scorer(h12)
            

            # print("mc_logits: ", mc_logits.shape)
            reshaped_logits_mrc = mc_logits.view(-1, num_choices)
            # print("reshaped_logits_mrc: ", reshaped_logits_mrc.shape)
            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            # print("len: ", len(outputs))
            mse_loss = calc_kl_div(reshaped_logits_mrc, explanation_classification_scores, temperature=2.0)
            
            outputs = outputs + outputs_mrc 

        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs
            loss_fct_explanation = CrossEntropyLoss()
            mse_loss += loss_fct_explanation(explanation_classification_scores, labels_mrc)
            outputs = (mse_loss, ) + outputs
        
        # 三个loss
        # 第一个是带解释的分类softmax损失+两个分布的KL loss（temperature=2.0)
        # 第二个是5个选项分数softmax的分类损失
        # 第三个是解释的生成损失
        # mse mrc lm
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)


class NazaOneLMAddMrcHeadModel(BertPreTrainedModel):
    def __init__(self, config):
        # config.update({'is_decoder': True})
        super().__init__(config)
        # assert config.is_decoder, "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`."
        self.num_choices = 5
        config.update({'is_decoder': True})
        self.bert = BertModel(config)
        
        self.cls = BertOnlyMLMHead(config)
        # self.dropout = nn.Dropout(0.1)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        
        config.summary_type = "cls_index"
        # config.summary_type = "first"
        config.summary_proj_to_labels = True
        config.summary_use_proj = True
        config.summary_first_dropout = 0.15
        config.num_labels = 1
        self.config = config
        self.multiple_choice_head = SequenceSummary(config)
        # self.att_merge = AttentionMerge(config.hidden_size, attention_size=1024, dropout_prob=0.1)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.scorer = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hidden_size, 1)
        # )
        config.num_labels = 5
        self.explanation_head = SequenceSummary(config)
        # self.att_merge_2 = AttentionMerge(config.hidden_size, attention_size=1024, dropout_prob=0.1)
        # self.scorer_2 = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(config.hidden_size, self.num_choices)
        # )
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_mrc=None,
        attention_mask_mrc=None,
        token_type_ids_mrc=None,
        position_ids_mrc=None,
        head_mask_mrc=None,
        inputs_embeds_mrc=None,
        labels=None,
        labels_mrc=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        encoder_hidden_states_mrc=None,
        encoder_attention_mask_mrc=None,
        output_attentions_mrc=None,
        output_hidden_states_mrc=None,
        **kwargs
    ):
        '''
        - [CLS] context [SEP] choice_1 [SEP]
        - [CLS] context [SEP] choice_2 [SEP]
        - [CLS] context [SEP] choice_3 [SEP]
        - [CLS] context [SEP] choice_4 [SEP]
        - [CLS] context [SEP] choice_5 [SEP]
        '''
        # print("input_ids: ###", input_ids.shape)
        assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
        #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
        num_choices = self.num_choices
        return_dict = self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if attention_mask is not None:
            pre = torch.sum(attention_mask, axis=-1) 
            attention_length = pre - torch.ones_like(pre)
        else:
            attention_length = None
        # print(attention_mask)
        # print(attention_length)
        # exit(0)
        sequence_output = outputs[0]
        # sequence_output = sequence_output * (self.config.hidden_size ** -0.5)
        prediction_scores = self.cls(sequence_output)
        # prediction_scores = None
        # explanation_classification_scores = self.tg_classifier(outputs[1])

        explanation_classification_scores = self.explanation_head(sequence_output, attention_length).squeeze(-1)
        # h_exp = self.att_merge_2(outputs[0], attention_mask)
        # explanation_classification_scores = self.scorer_2(h_exp)
            
        explanation_classification_scores = explanation_classification_scores.view(-1, num_choices)
        # print("explanation_classification_scores: ", explanation_classification_scores.shape)
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if labels is None and labels_mrc is None:
            return CausalLMOutputWithPast(logits=prediction_scores)
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction='sum')
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            
            # ltr_lm_loss = None
            outputs = (ltr_lm_loss,) + outputs

        input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
        attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
        token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
        position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
        inputs_embeds_mrc = (
            inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
            if inputs_embeds_mrc is not None
            else None
        )
        mrc_outputs = self.bert(
            input_ids_mrc,
            attention_mask=attention_mask_mrc,
            token_type_ids=token_type_ids_mrc,
            position_ids=position_ids_mrc,
            head_mask=head_mask_mrc,
            inputs_embeds=inputs_embeds_mrc,
            # encoder_hidden_states=encoder_hidden_states_mrc,
            encoder_attention_mask=attention_mask_mrc,
            output_attentions=output_attentions_mrc,
            output_hidden_states=output_hidden_states_mrc,
            return_dict=return_dict,
        )
        hidden_states = mrc_outputs[0]
        pre = torch.sum(attention_mask_mrc, axis=-1) 
        attention_mrc_length = pre - torch.ones_like(pre)
        # print(attention_mrc_length)
        if labels_mrc is not None:
            # pooled_output_mrc = mrc_outputs[1]
            # pooled_output_mrc = self.dropout(pooled_output_mrc)
            # logits_mrc = self.classifier(pooled_output_mrc)
            # reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
            # print(hidden_states.shape)
            mc_logits = self.multiple_choice_head(hidden_states, attention_mrc_length).squeeze(-1)

            # h12 = self.att_merge(mrc_outputs[0], attention_mask_mrc)
            # mc_logits = self.scorer(h12)
            

            # print("mc_logits: ", mc_logits.shape)
            reshaped_logits_mrc = mc_logits.view(-1, num_choices)
            # print("reshaped_logits_mrc: ", reshaped_logits_mrc.shape)
            outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
            # print("len: ", len(outputs))
            
            # mse_loss = calc_mse_loss(reshaped_logits_mrc, explanation_classification_scores)
            
            outputs = outputs + outputs_mrc 
        if labels_mrc is not None:
            loss_fct_mrc = CrossEntropyLoss()
            loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
            outputs =  (loss_mrc,)  + outputs
            loss_fct_explanation = CrossEntropyLoss()
            mse_loss = loss_fct_explanation(explanation_classification_scores, labels_mrc)
            mse_loss += 0.1*calc_kl_div(reshaped_logits_mrc, explanation_classification_scores, temperature=1.0)
            outputs = (mse_loss, ) + outputs
        
        # 三个loss
        # 第一个是带解释的分类softmax损失+两个分布的KL loss（temperature=2.0)
        # 第二个是5个选项分数softmax的分类损失
        # 第三个是解释的生成损失
        # mse mrc lm
        return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        # input_shape = input_ids.shape
        # effective_batch_size = input_shape[0]

        # #  add a dummy token
        # assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        # attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # dummy_token = torch.full(
        #     (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        # )
        # # input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": None, \
                    "input_ids_mrc": model_kwargs["input_ids_mrc"], "attention_mask_mrc": model_kwargs["attention_mask_mrc"], "token_type_ids_mrc": model_kwargs["token_type_ids_mrc"]}
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    

# class NazaOneLMAddMrcHeadModel_only_test_mrc(BertPreTrainedModel):
#     def __init__(self, config):
#         # config.update({'is_decoder': True})
#         super().__init__(config)
#         # assert config.is_decoder, "If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`."
#         self.num_choices = 5
#         config.update({'is_decoder': True})
#         self.bert = BertModel(config)
#         # self.cls = BertOnlyMLMHead(config)
#         # self.dropout = nn.Dropout(0.1)
#         # self.classifier = nn.Linear(config.hidden_size, 1)
        
#         config.summary_type = "cls_index"
#         # # config.summary_type = "first"
#         config.summary_proj_to_labels = True
#         config.summary_use_proj = True
#         config.summary_first_dropout = 0.1
#         config.num_labels = 1
#         self.config = config
#         self.multiple_choice_head = SequenceSummary(config)
#         # # self.att_merge = AttentionMerge(config.hidden_size, attention_size=1024, dropout_prob=0.1)
#         # # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         # # self.scorer = nn.Sequential(
#         # #     nn.Dropout(0.1),
#         # #     nn.Linear(config.hidden_size, 1)
#         # # )
#         # config.num_labels = 5
#         # self.explanation_head = SequenceSummary(config)
#         # # self.att_merge_2 = AttentionMerge(config.hidden_size, attention_size=1024, dropout_prob=0.1)
#         # # self.scorer_2 = nn.Sequential(
#         # #     nn.Dropout(0.1),
#         # #     nn.Linear(config.hidden_size, self.num_choices)
#         # # )
#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         input_ids_mrc=None,
#         attention_mask_mrc=None,
#         token_type_ids_mrc=None,
#         position_ids_mrc=None,
#         head_mask_mrc=None,
#         inputs_embeds_mrc=None,
#         labels=None,
#         labels_mrc=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         encoder_hidden_states_mrc=None,
#         encoder_attention_mask_mrc=None,
#         output_attentions_mrc=None,
#         output_hidden_states_mrc=None,
#         **kwargs
#     ):
#         '''
#         - [CLS] context [SEP] choice_1 [SEP]
#         - [CLS] context [SEP] choice_2 [SEP]
#         - [CLS] context [SEP] choice_3 [SEP]
#         - [CLS] context [SEP] choice_4 [SEP]
#         - [CLS] context [SEP] choice_5 [SEP]
#         '''
#         assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
#         #num_choices = input_ids_mrc.shape[1] if input_ids_mrc is not None else inputs_embeds_mrc.shape[1]
#         num_choices = self.num_choices
#         # outputs = self.bert(
#         #     input_ids,
#         #     attention_mask=attention_mask,
#         #     token_type_ids=token_type_ids,
#         #     position_ids=position_ids,
#         #     head_mask=head_mask,
#         #     inputs_embeds=inputs_embeds,
#         #     # encoder_hidden_states=encoder_hidden_states,
#         #     encoder_attention_mask=attention_mask,
#         #     output_attentions=output_attentions,
#         #     output_hidden_states=output_hidden_states,
#         # )
#         # if attention_mask is not None:
#         #     pre = torch.sum(attention_mask, axis=-1) 
#         #     attention_length = pre - torch.ones_like(pre)
#         # else:
#         #     attention_length = None
#         # print(attention_mask)
#         # print(attention_length)
#         # exit(0)
#         # sequence_output = outputs[0]
#         # prediction_scores = self.cls(sequence_output)
#         prediction_scores = None
#         # explanation_classification_scores = self.tg_classifier(outputs[1])

#         # explanation_classification_scores = self.explanation_head(sequence_output, attention_length).squeeze(-1)
#         # h_exp = self.att_merge_2(outputs[0], attention_mask)
#         # explanation_classification_scores = self.scorer_2(h_exp)
            
#         # explanation_classification_scores = explanation_classification_scores.view(-1, num_choices)
#         # print("explanation_classification_scores: ", explanation_classification_scores.shape)
#         outputs = (prediction_scores,) #+ outputs[2:]  # Add hidden states and attention if they are here
        
#         if labels is not None:
#             # we are doing next-token prediction; shift prediction scores and input ids by one
            
#             # prediction_scores = prediction_scores[:, :-1, :].contiguous()
#             labels = labels[:, 1:].contiguous()
#             # loss_fct = CrossEntropyLoss()
#             # ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            
#             ltr_lm_loss = None
#             outputs = (ltr_lm_loss,) + outputs

#         input_ids_mrc = input_ids_mrc.view(-1, input_ids_mrc.size(-1)) if input_ids_mrc is not None else None
#         attention_mask_mrc = attention_mask_mrc.view(-1, attention_mask_mrc.size(-1)) if attention_mask_mrc is not None else None
#         token_type_ids_mrc = token_type_ids_mrc.view(-1, token_type_ids_mrc.size(-1)) if token_type_ids_mrc is not None else None
#         position_ids_mrc = position_ids_mrc.view(-1, position_ids_mrc.size(-1)) if position_ids_mrc is not None else None
#         inputs_embeds_mrc = (
#             inputs_embeds_mrc.view(-1, inputs_embeds_mrc.size(-2), inputs_embeds_mrc.size(-1))
#             if inputs_embeds_mrc is not None
#             else None
#         )
#         mrc_outputs = self.bert(
#             input_ids_mrc,
#             attention_mask=attention_mask_mrc,
#             token_type_ids=token_type_ids_mrc,
#             position_ids=position_ids_mrc,
#             head_mask=head_mask_mrc,
#             inputs_embeds=inputs_embeds_mrc,
#             # encoder_hidden_states=encoder_hidden_states_mrc,
#             encoder_attention_mask=attention_mask_mrc,
#             output_attentions=output_attentions_mrc,
#             output_hidden_states=output_hidden_states_mrc,
#         )
#         hidden_states = mrc_outputs[0]
#         pre = torch.sum(attention_mask_mrc, axis=-1) 
#         attention_mrc_length = pre - torch.ones_like(pre)
#         # print(attention_mrc_length)
#         if labels_mrc is not None:
#             # pooled_output_mrc = mrc_outputs[1]
#             # pooled_output_mrc = self.dropout(pooled_output_mrc)
#             # logits_mrc = self.classifier(pooled_output_mrc)
#             # reshaped_logits_mrc = logits_mrc.view(-1, num_choices)
#             # print(hidden_states.shape)
#             mc_logits = self.multiple_choice_head(hidden_states, attention_mrc_length).squeeze(-1)

#             # h12 = self.att_merge(mrc_outputs[0], attention_mask_mrc)
#             # mc_logits = self.scorer(h12)
            

#             # print("mc_logits: ", mc_logits.shape)
#             reshaped_logits_mrc = mc_logits.view(-1, num_choices)
#             # print("reshaped_logits_mrc: ", reshaped_logits_mrc.shape)
#             outputs_mrc = (reshaped_logits_mrc,) + mrc_outputs[2:]  # add hidden states and attention if they are here
#             # print("len: ", len(outputs))
            
#             # mse_loss = calc_mse_loss(reshaped_logits_mrc, explanation_classification_scores)
            
#             outputs = outputs + outputs_mrc 
#         if labels_mrc is not None:
#             loss_fct_mrc = CrossEntropyLoss()
#             loss_mrc = loss_fct_mrc(reshaped_logits_mrc, labels_mrc)
#             outputs =  (loss_mrc,)  + outputs
#             # loss_fct_explanation = CrossEntropyLoss()
#             # mse_loss = loss_fct_explanation(explanation_classification_scores, labels_mrc)
#             # mse_loss += 0.1*calc_kl_div(reshaped_logits_mrc, explanation_classification_scores, temperature=1.0)
#             outputs = (loss_mrc, ) + outputs
        
#         # 三个loss
#         # 第一个是带解释的分类softmax损失+两个分布的KL loss（temperature=2.0)
#         # 第二个是5个选项分数softmax的分类损失
#         # 第三个是解释的生成损失
#         # mse mrc lm
#         return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)
    
#     def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
#         input_shape = input_ids.shape
#         effective_batch_size = input_shape[0]

#         #  add a dummy token
#         assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
#         attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
#         dummy_token = torch.full(
#             (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
#         )
#         input_ids = torch.cat([input_ids, dummy_token], dim=1)

#         return {"input_ids": input_ids, "attention_mask": attention_mask}


# class NazaOneLMAddMrcHeadModel_r2o(BertPreTrainedModel):
#     def __init__(self, config):
#         # config.update({'is_decoder': True})
#         super().__init__(config)
#         self.num_choices = 5
#         config.update({'is_decoder': False})
#         self.bert = BertModel(config) 
#         self.dropout = nn.Dropout(0.1)
#         self.tg_classifier = nn.Linear(config.hidden_size, 5)
#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         input_ids_mrc=None,
#         attention_mask_mrc=None,
#         token_type_ids_mrc=None,
#         position_ids_mrc=None,
#         head_mask_mrc=None,
#         inputs_embeds_mrc=None,
#         labels=None,
#         labels_mrc=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         encoder_hidden_states_mrc=None,
#         encoder_attention_mask_mrc=None,
#         output_attentions_mrc=None,
#         output_hidden_states_mrc=None,
#         **kwargs
#     ):
#         assert ((labels is not None) and (labels_mrc is not None)) or ((labels is None) and (labels_mrc is None))
#         num_choices = self.num_choices
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             # encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#         )
#         if attention_mask is not None:
#             pre = torch.sum(attention_mask, axis=-1) 
#             attention_length = pre - torch.ones_like(pre)
#         else:
#             attention_length = None
#         sequence_output = outputs[0]
#         prediction_scores = None
#         pooled_output = self.dropout(outputs[1])
#         explanation_classification_scores = self.tg_classifier(pooled_output)
            
#         explanation_classification_scores = explanation_classification_scores.view(-1, num_choices)
#         # print("explanation_classification_scores: ", explanation_classification_scores.shape)
#         outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

#         if labels_mrc is not None:
#             outputs_mrc = (explanation_classification_scores,)
#             outputs = outputs + outputs_mrc 
#         if labels_mrc is not None:
#             loss_fct_explanation = CrossEntropyLoss()
#             mse_loss = loss_fct_explanation(explanation_classification_scores, labels_mrc)
#             loss_mrc = mse_loss
#             outputs =  (None,)  + outputs
#             outputs =  (loss_mrc,)  + outputs
#             outputs = (mse_loss, ) + outputs
            
        
#         # 三个loss
#         # 第一个是带解释的分类softmax损失+两个分布的KL loss（temperature=2.0)
#         # 第二个是5个选项分数softmax的分类损失
#         # 第三个是解释的生成损失
#         # mse mrc lm
#         return outputs # (loss_mrc, ),  (ltr_lm_loss, ), prediction_scores, (hidden_states), (attentions), reshaped_logits, (hidden_states), (attentions)
    
    