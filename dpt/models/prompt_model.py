import torch
from torch import nn
from torch.nn import Parameter
from transformers import BertModel, BertPreTrainedModel, ElectraModel, ElectraForPreTraining, ElectraPreTrainedModel
from DataProcessorClass.data_processors import data_processors
from models.params import params

class BeginWithLabels(ElectraPreTrainedModel):
    name: str = "begin_with_labels"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraForPreTraining.from_pretrained(
            params[model_args.dpt_path],
            config=config,
            cache_dir=model_args.folder_cache
        )

        self.tokenizer = tokenizer
        self.labels = data_processors[data_args.task_name].receiveLblTxt()
        self.prompts = ["labels", ":"]
        self.loss_func = model_args.func_loss
        self.bias = Parameter(torch.zeros(self.num_labels))


        lti = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(lb)) for i, lb in enumerate(self.labels)]
        self.label_postions = []
        self.cls_tokens = []
        flti = []
        i = len(self.prompts)
        print(lti)
        for tids in lti:
            if type(tids) is not list:
                tids = [tids]
            self.label_postions.append(i)
            for tid in tids:
                self.cls_tokens.append("[unused{}]".format(i + 100))
                flti.append(tid)
                i += 1
        cti = self.tokenizer.convert_tokens_to_ids(self.cls_tokens)
        for ci, li in zip(cti, flti):
            with torch.no_grad():
                val = self.electra.electra.embeddings.word_embeddings.weight.data[li]
                self.electra.electra.embeddings.word_embeddings.weight[ci].copy_(val)
        self.cls_tokens = self.prompts + self.cls_tokens
        print(self.label_postions, self.cls_tokens, self.tokenizer.convert_ids_to_tokens(flti))

    def tokenize(self, data, **kwargs):
        max_len = kwargs.pop("max_length", 512)
        padding = kwargs.pop("padding", None)
        truncation = kwargs.pop("truncation", False)

        # construct label tokens
        label_tokens = self.cls_tokens]
        label_positions = self.label_postions
        label_tokens = label_tokens + ["[SEP]"]

        dic = label_tokenizer(data, self.tokenizer, label_tokens, max_len, label_positions)
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

        outputs = self.electra(
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
        raw_logits = outputs[0]
        cls_logits = []
        if len(raw_logits.size()) == 1:
            raw_logits = raw_logits.unsqueeze(0)
        batch_size = raw_logits.size(0)
        for i in range(self.num_labels):
            i_cls_logits = raw_logits[torch.arange(batch_size), positions[:, i]]
            cls_logits.append(i_cls_logits.unsqueeze(1))
        logits = torch.cat(cls_logits, 1) + self.bias

        loss = None
        if labels is not None:
            targets = torch.zeros_like(logits.view(-1, self.num_labels), device=logits.device).scatter_(1, labels.view(-1, 1),1)
            loss_fct = nn.BCELoss()
            logits = 1 - torch.sigmoid(logits)
            loss = loss_fct(logits.view(-1, self.num_labels), targets)
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class BeginWithLabelsAndDownline(ElectraPreTrainedModel):
    name: str = "begin_with_labels_and_downline"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraForPreTraining.from_pretrained(
            params[model_args.dpt_path],
            config=config,
            cache_dir=model_args.folder_cache
        )

        self.tokenizer = tokenizer
        self.labels = data_processors[data_args.task_name].receiveLblTxt()
        self.prompts = ["labels", ":", "_"]
        self.loss_func = model_args.func_loss
        self.bias = Parameter(torch.zeros(self.num_labels))


        lti = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(lb)) for i, lb in enumerate(self.labels)]
        self.label_postions = []
        self.cls_tokens = []
        flti = []
        i = len(self.prompts)
        print(lti)
        for tids in lti:
            if type(tids) is not list:
                tids = [tids]
            self.label_postions.append(i)
            for tid in tids:
                self.cls_tokens.append("[unused{}]".format(i + 100))
                flti.append(tid)
                i += 1
        cli = self.tokenizer.convert_tokens_to_ids(self.cls_tokens)
        for ci, li in zip(cli, flti):
            with torch.no_grad():
                val = self.electra.electra.embeddings.word_embeddings.weight.data[li]
                self.electra.electra.embeddings.word_embeddings.weight[ci].copy_(val)
        self.cls_tokens = self.prompts + self.cls_tokens
        print(self.label_postions, self.cls_tokens, self.tokenizer.convert_ids_to_tokens(flti))

    def tokenize(self, data, **kwargs):
        max_len = kwargs.pop("max_length", 512)
        padding = kwargs.pop("padding", None)
        truncation = kwargs.pop("truncation", False)

        label_tokens = self.cls_tokens
        label_positions = self.label_postions
        label_tokens = label_tokens + ["[SEP]"]

        dic = label_tokenizer(data, self.tokenizer, label_tokens, max_len, label_positions)
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

        outputs = self.electra(
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
        raw_logits = outputs[0]
        cls_logits = []
        if len(raw_logits.size()) == 1:
            raw_logits = raw_logits.unsqueeze(0)
        batch_size = raw_logits.size(0)
        for i in range(self.num_labels):
            i_cls_logits = raw_logits[torch.arange(batch_size), positions[:, i]]
            cls_logits.append(i_cls_logits.unsqueeze(1))
        logits = torch.cat(cls_logits, 1) + self.bias

        loss = None
        if labels is not None:
            targets = torch.zeros_like(logits.view(-1, self.num_labels), device=logits.device).scatter_(1, labels.view(-1, 1),1)
            loss_fct = nn.BCELoss()
            logits = 1 - torch.sigmoid(logits)
            loss = loss_fct(logits.view(-1, self.num_labels), targets)
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class BeginWithSentiment(ElectraPreTrainedModel):
    name: str = "begin_with_sentiment"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraForPreTraining.from_pretrained(
            params[model_args.dpt_path],
            config=config,
            cache_dir=model_args.folder_cache
        )

        self.tokenizer = tokenizer
        self.labels = data_processors[data_args.task_name].receiveLblTxt()
        self.prompts = ["sentiment", ":"]
        self.loss_func = model_args.func_loss
        self.bias = Parameter(torch.zeros(self.num_labels))


        lti = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(lb)) for i, lb in enumerate(self.labels)]
        self.label_postions = []
        self.cls_tokens = []
        flti = []
        i = len(self.prompts)
        print(lti)
        for tids in lti:
            if type(tids) is not list:
                tids = [tids]
            self.label_postions.append(i)
            for tid in tids:
                self.cls_tokens.append("[unused{}]".format(i + 100))
                flti.append(tid)
                i += 1
        cli = self.tokenizer.convert_tokens_to_ids(self.cls_tokens)
        for ci, li in zip(cli, flti):
            with torch.no_grad():
                val = self.electra.electra.embeddings.word_embeddings.weight.data[li]
                self.electra.electra.embeddings.word_embeddings.weight[ci].copy_(val)
        self.cls_tokens = self.prompts + self.cls_tokens
        print(self.label_postions, self.cls_tokens, self.tokenizer.convert_ids_to_tokens(flti))

    def tokenize(self, data, **kwargs):
        max_len = kwargs.pop("max_length", 512)
        padding = kwargs.pop("padding", None)
        truncation = kwargs.pop("truncation", False)

        label_tokens = self.cls_tokens
        label_positions = self.label_postions
        label_tokens = label_tokens + ["[SEP]"]

        dic = label_tokenizer(data, self.tokenizer, label_tokens, max_len, label_positions)
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

        outputs = self.electra(
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
        raw_logits = outputs[0]
        cls_logits = []
        if len(raw_logits.size()) == 1:
            raw_logits = raw_logits.unsqueeze(0)
        batch_size = raw_logits.size(0)
        for i in range(self.num_labels):
            i_cls_logits = raw_logits[torch.arange(batch_size), positions[:, i]]
            cls_logits.append(i_cls_logits.unsqueeze(1))
        logits = torch.cat(cls_logits, 1) + self.bias

        loss = None
        if labels is not None:
            targets = torch.zeros_like(logits.view(-1, self.num_labels), device=logits.device).scatter_(1, labels.view(-1, 1),1)
            loss_fct = nn.BCELoss()
            logits = 1 - torch.sigmoid(logits)
            loss = loss_fct(logits.view(-1, self.num_labels), targets)
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class BeginWithSentimentAndDownline(ElectraPreTrainedModel):
    name: str = "begin_with_sentiment_and_downline"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraForPreTraining.from_pretrained(
            params[model_args.dpt_path],
            config=config,
            cache_dir=model_args.folder_cache
        )

        self.tokenizer = tokenizer
        self.labels = data_processors[data_args.task_name].receiveLblTxt()
        self.prompts = ["sentiment", ":", "_"] # sst2
        self.loss_func = model_args.func_loss
        self.bias = Parameter(torch.zeros(self.num_labels))


        lti = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(lb)) for i, lb in enumerate(self.labels)]
        
        self.label_postions = []
        self.cls_tokens = []
        flti = []
        i = len(self.prompts)
        print(lti)
        for tids in lti:
            if type(tids) is not list:
                tids = [tids]
            self.label_postions.append(i)
            for tid in tids:
                self.cls_tokens.append("[unused{}]".format(i + 100))
                flti.append(tid)
                i += 1
        cli = self.tokenizer.convert_tokens_to_ids(self.cls_tokens)
        for ci, li in zip(cli, flti):
            with torch.no_grad():
                val = self.electra.electra.embeddings.word_embeddings.weight.data[li]
                self.electra.electra.embeddings.word_embeddings.weight[ci].copy_(val)
        self.cls_tokens = self.prompts + self.cls_tokens
        print(self.label_postions, self.cls_tokens, self.tokenizer.convert_ids_to_tokens(flti))

    def tokenize(self, data, **kwargs):
        max_len = kwargs.pop("max_length", 512)
        padding = kwargs.pop("padding", None)
        truncation = kwargs.pop("truncation", False)

        label_tokens = self.cls_tokens
        label_positions = self.label_postions
        label_tokens = label_tokens + ["[SEP]"]

        dic = label_tokenizer(data, self.tokenizer, label_tokens, max_len, label_positions)
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

        outputs = self.electra(
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
        raw_logits = outputs[0]
        cls_logits = []
        if len(raw_logits.size()) == 1:
            raw_logits = raw_logits.unsqueeze(0)
        batch_size = raw_logits.size(0)
        for i in range(self.num_labels):
            i_cls_logits = raw_logits[torch.arange(batch_size), positions[:, i]]
            cls_logits.append(i_cls_logits.unsqueeze(1))
        logits = torch.cat(cls_logits, 1) + self.bias

        loss = None
        if labels is not None:
            targets = torch.zeros_like(logits.view(-1, self.num_labels), device=logits.device).scatter_(1, labels.view(-1, 1),1)
            loss_fct = nn.BCELoss()
            logits = 1 - torch.sigmoid(logits)
            loss = loss_fct(logits.view(-1, self.num_labels), targets)
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

