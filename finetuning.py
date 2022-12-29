import logging
import os
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import wandb

import datasets
import numpy as np
from datasets import load_dataset
from dataHelper import get_dataset

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version



logger = logging.getLogger(__name__)

        
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "(Useless) The name of the task to train on"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_my_tokenizer: Optional[str] = field(
        default=False, metadata={"help": "(Useless) Use my tokenizer or not"}
    )         
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
        
def main():
    '''
        initialize logging, seed, argparse...
    '''
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    wandb.init(project="YOUR_PROJECT", entity="YOUR_ENTITY", name=training_args.output_dir)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    '''
        load datasets
    '''
    raw_datasets = get_dataset(data_args.dataset_name)
    label_list = raw_datasets["train"].unique("labels")
    num_labels = len(label_list)

    '''
        load models
    '''
   
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config
    )
    
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    arg_file = os.path.join(training_args.output_dir, "mytraining_args.txt")
    if not os.path.exists(arg_file):
        os.mknod(arg_file)
    with open(arg_file, 'w') as f0:
        print(training_args, file=f0)
        print(f'use_my_tokenizer:{model_args.use_my_tokenizer}-{model.get_input_embeddings().weight.shape[0]}', file=f0)
        f0.close()    

    '''
        process datasets and build up datacollator
    '''
    padding = "max_length"
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    def preprocess_function(example):
        result = tokenizer(example["text"], padding=padding, max_length=max_seq_length, truncation=True)
        result["labels"] = example["labels"]
        return result

    raw_datasets = raw_datasets.map(preprocess_function, batched=True)

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    mytrain_dataset = raw_datasets["train"].shuffle(seed=training_args.seed)

    if "test" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    mypredict_dataset = raw_datasets["test"]

    if "dev" not in raw_datasets:
        raise ValueError("--do_eval requires a dev dataset")
    mydev_dataset = raw_datasets["dev"]

    metrics_lst = ["accuracy", "f1"]
    metric = {x: evaluate.load(x) for x in metrics_lst}

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = {"accuracy": metric["accuracy"].compute(predictions=preds, references=p.label_ids)["accuracy"]}
        result["micro_f1"] = metric["f1"].compute(predictions=preds, references=p.label_ids, average='micro')["f1"]
        result["macro_f1"] = metric["f1"].compute(predictions=preds, references=p.label_ids, average='macro')["f1"]
        return result

    data_collator = default_data_collator

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mytrain_dataset,
        eval_dataset=mydev_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    ) # build up trainer

    # training!
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            len(mytrain_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(mytrain_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()      
        
    if training_args.do_predict:
        logger.info("*** Predict ***")
        if data_args.dataset_name[-2:]=='fs':
            prediction_output = trainer.predict(mypredict_dataset, metric_key_prefix="predict")
            trainer.log_metrics("predict", prediction_output.metrics)
            trainer.save_metrics("predict", prediction_output.metrics)  
            
        else:
            # Choose best model during training
            best_f1 = 0
            best_ckpt = None
            best_metrics = None


            checkpoint_lst = [os.path.join(training_args.output_dir, x) 
                              for x in os.listdir(training_args.output_dir) 
                              if os.path.isdir(os.path.join(training_args.output_dir, x)) and x[0] != '.']

            for pred_ckpt in checkpoint_lst:
                logger.info(f"testing {pred_ckpt}")
                tmp_config = AutoConfig.from_pretrained(
                    pred_ckpt,
                    num_labels=num_labels
                )

                tmp_tokenizer = AutoTokenizer.from_pretrained(pred_ckpt)

                tmp_model = AutoModelForSequenceClassification.from_pretrained(
                    pred_ckpt,
                    from_tf=bool(".ckpt" in pred_ckpt),
                    config=tmp_config
                ) 

                tmp_trainer = Trainer(
                    model=tmp_model,
                    args=training_args,
                    train_dataset=mytrain_dataset,
                    compute_metrics=compute_metrics,
                    tokenizer=tmp_tokenizer,
                    data_collator=data_collator,
                ) 

                tmp_prediction_output = tmp_trainer.predict(mypredict_dataset, metric_key_prefix="predict")
                tmp_path = os.path.join(pred_ckpt, "pred_results.json")
                with open(tmp_path, "w") as ft:
                    json.dump(tmp_prediction_output.metrics, ft, indent=4, sort_keys=True)
                    ft.close()

                if tmp_prediction_output.metrics['predict_macro_f1'] > best_f1:
                    best_f1 = tmp_prediction_output.metrics['predict_macro_f1']
                    best_ckpt = pred_ckpt
                    logger.info(f"best_ckpt: {best_ckpt}, best_f1: {best_f1}")
                    best_metrics = tmp_prediction_output.metrics

            final_path = os.path.join(training_args.output_dir, "final_results.json")
            with open(final_path, "w") as ff:
                json.dump(best_metrics, ff, indent=4, sort_keys=True)
                ff.close()
        

if __name__=='__main__':
    main()
