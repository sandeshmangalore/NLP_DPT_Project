import os
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from transformers import EvalPrediction
from DataProcessorClass.dataset_class import DataSetClass
from DataProcessorClass.dataset_class import TrainDataArgs as TDArgs
from transformers import HfArgumentParser, Trainer, TrainingArguments, glue_compute_metrics, set_seed
from DataProcessorClass.dataset_class import data_modes
from models.build import build_model


lgr = logging.getLogger(__name__)

def metric_acc(res, actual):
    return (res == actual).mean()


def metric_avg_acc(res, actual):
    kv = {}
    for i, j in zip(res, actual):
        j = int(j)
        if j not in kv:
            kv[j] = [0, 0]
        kv[j][1] += 1
        kv[j][0] += i==j
    kv = {a: b[0]/b[1] for a, b in kv.items()}
    acc_avg = sum(kv.values())/len(kv)
    kv["acc_avg"] = acc_avg
    kv = {a: round(b, 5) for a, b in kv.items()}
    return acc_avg


@dataclass
class DptArgs:
    dpt_path: str = field()
    pretrained_model: str = field(default=None)
    config_name_s: Optional[str] = field(default=None)
    tokenizer_name_s: Optional[str] = field(default=None)
    folder_cache: Optional[str] = field(default=None)
    func_loss: Optional[str] = field(default="CE")


def main():

    hfap = HfArgumentParser((DptArgs, TDArgs, TrainingArguments))

    mArguments, dArguments, tArguments = hfap.parse_args_into_dataclasses()

    if (os.path.exists(tArguments.output_dir) and os.listdir(tArguments.output_dir)
        and tArguments.do_train and not tArguments.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({tArguments.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    set_seed(tArguments.seed)
    model = build_model(mArguments, dArguments)

    data_tr = (
        DataSetClass(dArguments, tokenizer=model.tokenize, cache_dir=dArguments.data_cached_dir) if tArguments.do_train else None
    )
    data_ev = (
        DataSetClass(dArguments, tokenizer=model.tokenize, mode="dev", cache_dir=dArguments.data_cached_dir)
        if tArguments.do_eval
        else None
    )
    data_ts = (
        DataSetClass(dArguments, tokenizer=model.tokenize, mode="test", cache_dir=dArguments.data_cached_dir)
    )

    mode_out = data_modes[dArguments.task_name]
    def metrics_build(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def metrics_compute(p: EvalPrediction):
            res = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if mode_out == "classification":
                res = np.argmax(res, axis=1)
            else:  # regression
                res = np.squeeze(res)
            return {"acc": metric_acc(res, p.label_ids), "avg_acc": metric_avg_acc(res, p.label_ids)}

        return metrics_compute

    trainer = Trainer(model=model,args=tArguments,train_dataset=data_tr,eval_dataset=data_ev,
        compute_metrics=metrics_build(dArguments.task_name),)

    if tArguments.do_train:
        trainer.train(
            model_path=mArguments.pretrained_model if mArguments.pretrained_model is not None and
                                                           os.path.isdir(mArguments.pretrained_model) else None
        )
        trainer.save_model()

    resl = {}
    if tArguments.do_eval:
        lgr.info("*** Evaluate ***")

        eds = [data_ts]

        for ed in eds:
            trainer.compute_metrics = metrics_build(data_ev.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=ed)

            eval_op_file = os.path.join(tArguments.output_dir, f"resl_{data_ev.args.task_name}.txt")
            with open(eval_op_file, "w") as writer:
                lgr.info("***** Eval results {} *****".format(data_ev.args.task_name))
                for k, v in eval_result.items():
                    lgr.info("  %s = %s", k, v)
                    writer.write("%s = %s\n" % (k, v))

            resl.update(eval_result)


    if tArguments.do_predict:
        logging.info("*** Test ***")
        tds = [data_ts]

        for td in tds:
            ops = trainer.predict(test_dataset=td).predictions
            if mode_out == "classification":
                ops = np.argmax(ops, axis=1)

            test_op_file = os.path.join(
                tArguments.output_dir, f"test_results_{data_ts.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(test_op_file, "w") as writer:
                    lgr.info("***** Test results {} *****".format(data_ts.args.task_name))
                    writer.write("index\tprediction\n")
                    for idx, i in enumerate(ops):
                        if mode_out == "regression":
                            writer.write("%d\t%3.3f\n" % (idx, i))
                        else:
                            i = data_ts.get_labels()[i]
                            writer.write("%d\t%s\n" % (idx, i))
    return resl


if __name__ == "__main__":
    main()