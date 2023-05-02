import os
import warnings

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, Parameter
from transformers import BertModel, BertPreTrainedModel, ElectraModel, ElectraForPreTraining, ElectraPreTrainedModel
from DataProcessorClass.data_processors import data_processors
from models.params import params


class Bert(BertPreTrainedModel):
    name: str = "bert"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel.from_pretrained(
            params[model_args.model_name_or_path],
            config=config,
            cache_dir=model_args.cache_dir
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.tokenizer = tokenizer
        self.labels = None

    def tokenize(self, data, **kwargs):
        kwargs.pop("labels")
        dic = self.tokenizer(data, **kwargs)
        return dic

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class FSBertWithLabelMarker(BertPreTrainedModel):
    name: str = "fs_bert_with_label_marker"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel.from_pretrained(
            params[model_args.model_name_or_path],
            config=config,
            cache_dir=model_args.cache_dir
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1, bias=False)

        self.tokenizer = tokenizer
        self.labels = None
        self.loss_func = model_args.loss_func
        self.bias = Parameter(torch.Tensor(self.num_labels))

    def tokenize(self, data, **kwargs):
        max_len = kwargs.pop("max_length", 512)
        padding = kwargs.pop("padding", None)
        truncation = kwargs.pop("truncation", False)

        labels = kwargs.pop("labels")
        label_tokens = [["[unused{}]".format(i + 100)] + self.tokenizer.tokenize(lb) for i, lb in enumerate(labels)]
        label_positions = [0] + [(len(t)) for t in label_tokens]
        for i, p in enumerate(label_positions):
            if i == 0:
                continue
            label_positions[i] = label_positions[i - 1] + p
        label_positions = label_positions[:-1]

        label_tokens = [item for item_list in label_tokens for item in item_list] + ["[SEP]"]

        dic = tokenize_all_data_with_labels(data, self.tokenizer, label_tokens, max_len, label_positions)
        return dic

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            positions=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        cls_logits = []
        batch_size = sequence_output.size(0)
        for i in range(self.num_labels):
            i_cls_embedding = sequence_output[torch.arange(batch_size), positions[:, i],:]
            i_cls_embedding = self.dropout(i_cls_embedding)
            i_cls_logits = self.classifier(i_cls_embedding)
            cls_logits.append(i_cls_logits)
        logits = torch.cat(cls_logits, 1) + self.bias

        loss = None
        if labels is not None:
            loss = loss_calculation(logits, labels, self.num_labels, self.loss_func)

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class FSBertWithLabelInit(BertPreTrainedModel):
    name: str = "fs_bert_with_label_init"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel.from_pretrained(
            params[model_args.model_name_or_path],
            config=config,
            cache_dir=model_args.cache_dir
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1, bias=False)

        self.tokenizer = tokenizer
        self.labels = data_processors[data_args.task_name].receiveLblTxt()
        self.loss_func = model_args.loss_func
        self.bias = Parameter(torch.zeros(self.num_labels))


        label_tokens_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(lb)) for i, lb in enumerate(self.labels)]
        # implementation 1
        self.label_postions = []
        self.cls_tokens = []
        flat_label_tokens_ids = []
        i = 0
        print(label_tokens_ids)
        for token_idxs in label_tokens_ids:
            if type(token_idxs) is not list:
                token_idxs = [token_idxs]
            self.label_postions.append(i)
            for tid in token_idxs:
                flat_label_tokens_ids.append(tid)
                self.cls_tokens.append("[unused{}]".format(i + 100))
                i += 1
        cls_token_ids = self.tokenizer.convert_tokens_to_ids(self.cls_tokens)
        for cls_id, label_id in zip(cls_token_ids, flat_label_tokens_ids):
            with torch.no_grad():
                val = self.bert.embeddings.word_embeddings.weight.data[label_id]
                self.bert.embeddings.word_embeddings.weight[cls_id].copy_(val)
        print(self.label_postions, self.cls_tokens, self.tokenizer.convert_ids_to_tokens(flat_label_tokens_ids))


    def tokenize(self, data, **kwargs):
        max_len = kwargs.pop("max_length", 512)
        padding = kwargs.pop("padding", None)
        truncation = kwargs.pop("truncation", False)

        label_tokens = self.cls_tokens
        label_positions = self.label_postions
        label_tokens = label_tokens + ["[SEP]"]

        # tokenize data
        dic = tokenize_all_data_with_labels(data, self.tokenizer, label_tokens, max_len, label_positions)
        return dic

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            positions=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        cls_logits = []
        batch_size = sequence_output.size(0)
        for i in range(self.num_labels):
            i_cls_embedding = sequence_output[torch.arange(batch_size), positions[:, i],:]
            i_cls_embedding = self.dropout(i_cls_embedding)
            i_cls_logits = self.classifier(i_cls_embedding)
            cls_logits.append(i_cls_logits)
        logits = torch.cat(cls_logits, 1) + self.bias

        l = None
        if labels is not None:
            l = loss_calculation(logits, labels, self.num_labels, self.loss_func)

        output = (logits,) + outputs[2:]
        return ((l,) + output) if l is not None else output


def loss_calculation(logits, labels, num_labels, loss_func):
    if num_labels == 1:
        lf = MSELoss()
        l = lf(logits.view(-1), labels.view(-1))
    elif lf == "CE":
        lf = CrossEntropyLoss()
        l = lf(logits.view(-1, num_labels), labels.view(-1))
    else:
        targets = torch.zeros_like(logits.view(-1, num_labels), device=logits.device).scatter_(1, labels.view(-1, 1),1)
        lf = nn.BCEWithLogitsLoss()
        l = lf(logits.view(-1, num_labels), targets)
    return l
