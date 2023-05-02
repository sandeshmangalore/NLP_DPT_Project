import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Callable, Dict

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers import PreTrainedTokenizerBase
from transformers import logging
from transformers.data.processors.utils import InputFeatures
from transformers import glue_convert_examples_to_features
from DataProcessorClass.data_processors import data_processors, data_modes

logger = logging.get_logger(__name__)


@dataclass
class TrainDataArgs:

    task_name: str = field()
    data_dir: str = field()
    data_cached_dir: str = field()
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)
    data_ratio: float = field(default=1.0)
    data_num: int = field(default=None)
    random_seed: int = field(default=0)

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class DataSetClass(Dataset):

    args: TrainDataArgs
    output_mode: str
    attributes: List[Dict]

    def __init__(
        self,
        args: TrainDataArgs,
        tokenizer: Callable,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = data_processors[args.task_name]()
        self.output_mode = data_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        file_cached_attributes = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )
        label_list = self.processor.receiveLbls()
        label_texts = self.processor.receiveLblTxt()
        self.label_list = label_list
        self.label_texts = label_texts

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        path_lock = file_cached_attributes + ".lock"
        with FileLock(path_lock):
            if os.path.exists(file_cached_attributes) and not args.overwrite_cache:
                start = time.time()
                self.attributes = torch.load(file_cached_attributes)
                logger.info(
                    f"Loading attributes from cached file {file_cached_attributes} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating attributes from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    samples = self.processor.receiveDevEx(args.data_dir)
                elif mode == Split.test:
                    samples = self.processor.receiveTsEx(args.data_dir)
                else:
                    samples = self.processor.receiveTrEx(args.data_dir)
                    samples = self.swcr(samples, self.args.data_ratio, data_num=self.args.data_num, random_seed=self.args.random_seed)
                if limit_length is not None:
                    samples = samples[:limit_length]
                self.attributes = transformExampleToFeature(
                    samples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    label_texts=label_texts,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.attributes, file_cached_attributes)
                logger.info(
                    "Saving attributes into cached file %s [took %.3f s]", file_cached_attributes, time.time() - start
                )

    def swcr(self, samples, ratio, data_num=None, random_seed=None):
        if 1 <= ratio < 2:
            return samples
        if ratio >= 2 and data_num is None:
            data_num = int(ratio)

        if random_seed is not None:
            random.seed(random_seed)

        d = {}
        for i, sample in enumerate(samples):
            lb = sample.label
            if lb not in d:
                d[lb] = []
            d[lb].append(i)
        for k, v in d.items():
            random.shuffle(v)
        # construct sampled data index list
        l = []
        for k, v in dic.items():
            v_len = len(v)
            n = data_num if data_num is not None else int(v_len*ratio)
            v = v[: n]
            l.extend(v)
        l = sorted(l)

        samples = [samples[i] for i in l]
        return samples

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, i) -> InputFeatures:
        return self.attributes[i]

    def get_labels(self):
        return self.label_list


def transformExampleToFeature(egs, tk, ml, task=None, ll=None, lt=None, om=None):
    if ml is None:
        ml = tk.max_len

    dictLabel = {l: j for j, l in enumerate(ll)}

    def egToLabel(eg) -> Union[int, float, None]:
        if eg.label is None:
            return None
        if om == "classification":
            return dictLabel[eg.label]
        elif om == "regression":
            return float(eg.label)
        raise KeyError(om)

    labels = [egToLabel(eg) for eg in egs]

    batch_encoding = tk(
        [(eg.text_a, eg.text_b) for eg in egs],
        max_length=ml,
        padding="ml",
        truncation=True,
        labels=lt,
    )

    attributes = []
    for i in range(len(egs)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        inputs['label'] = labels[i]
        attr = inputs
        attributes.append(attr)

    return attributes