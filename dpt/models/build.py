from transformers import AutoConfig, ElectraTokenizer, RobertaTokenizer, BertTokenizer
from DataProcessorClass.data_processors import data_nums
from models.bert_model import Bert, FSBertWithLabelMarker, FSBertWithLabelInit, params
from models.electra_model import Electra, FSElectraWithLabelMarker, FSElectraWithLabelItself, FSElectraWithLabelInit, FSElectraAddLabelItself, FSElectraAddLabelInit
from models.prompt_model import BeginWithLabels, BeginWithLabelsAndDownline, BeginWithSentiment, BeginWithSentimentAndDownline
import torch


map_model = {
    Bert.name: Bert,
    FSBertWithLabelMarker.name: FSBertWithLabelMarker,
    FSBertWithLabelInit.name: FSBertWithLabelInit,

    Electra.name: Electra,
    FSElectraWithLabelInit.name: FSElectraWithLabelInit,
    FSElectraWithLabelMarker.name: FSElectraWithLabelMarker,
    FSElectraWithLabelItself.name: FSElectraWithLabelItself,
    FSElectraAddLabelInit.name: FSElectraAddLabelInit,
    FSElectraAddLabelItself.name: FSElectraAddLabelItself,

    BeginWithLabels.name: BeginWithLabels,
    BeginWithLabelsAndDownline.name: BeginWithLabelsAndDownline,
    BeginWithSentiment.name: BeginWithSentiment,
    BeginWithSentimentAndDownline.name: BeginWithSentimentAndDownline,
}


def build_model(mArgs, dArgs):
    try:
        num_labels = data_nums[dArgs.task_name]
    except KeyError:
        raise ValueError("Dataset not found: %s" % (dArgs.task_name))

    config = AutoConfig.from_pretrained(
        mArgs.config_name if mArgs.config_name_s else params[mArgs.dpt_path],
        num_labels=num_labels,
        finetuning_task=dArgs.task_name,
        cache_dir=mArgs.folder_cache,
    )
    if 'roberta' in mArgs.dpt_path:
        tokenizer = RobertaTokenizer.from_pretrained(
            mArgs.tokenizer_name_s if mArgs.tokenizer_name_s else params[mArgs.dpt_path],
            cache_dir=mArgs.folder_cache,
        )
    elif 'bert' in mArgs.dpt_path:
        tokenizer = BertTokenizer.from_pretrained(
            mArgs.tokenizer_name_s if mArgs.tokenizer_name_s else params[mArgs.dpt_path],
            cache_dir=mArgs.folder_cache,
        )
    else:
        tokenizer = ElectraTokenizer.from_pretrained(
            mArgs.tokenizer_name_s if mArgs.tokenizer_name_s else params[mArgs.dpt_path],
            cache_dir=mArgs.folder_cache,
        )
    print(tokenizer)
    
    model = map_model[mArgs.dpt_path](mArgs, dArgs, config, tokenizer)
    if mArgs.pretrained_model:
        log = model.load_state_dict(torch.load(mArgs.pretrained_model))
        print(log)
    return model