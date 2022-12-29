# 基于扩充词表的特定领域后训练语言模型
## 介绍
现有大规模预训练语言模型一般关注通用语料，这些模型在下游特定领域的任务上表现欠佳。要提升在下游特定领域任务上的表现，常用方法是使用特定领域语料继续无监督训练模型，即后训练，再进行微调。本工作关注CS领域的NLP任务，基于RoBERTa，在实现掩码语言模型的后训练的基础上，尝试扩充词表后进行后训练，并实现两种后训练：在另外收集的特定领域语料(domain-adaptive pretraining, DAPT)、在下游任务的训练集上(task-adaptive pretraining, TAPT)进行无监督训练，实现中使用重新初始化word embedding、调整mask概率等手段。在本实验条件下，实验说明在后训练基础上，以本工作所用方法扩充词表的后训练对于CS领域NLP任务作用有限。本工作主要基于论文 ***Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*** [Paper](https://arxiv.org/abs/2004.10964) ，DAPT和TAPT来源于此。

## Requirements
|Name|Version|
|---------|-----|
|datasets|2.8.0|
|evaluate|0.4.0|
|jsonlines|3.1.0|
|transformers|4.25.1|
|torch|1.12.0|

另外，本工作使用 [Weights & Biases](https://wandb.ai/site) 记录训练过程。

首先需要自行安装PyTorch等，然后终端中运行如下命令安装所需要的dependencies。（如果不需要使用 Weights & Biases，请先删掉其中的 `pip install wandb` 和 `wandb login` ）（本工作在Linux上运行，对于不同的系统，请选择适合的方式运行`.sh`）
```
source requirements.sh
```

## 数据
CS领域语料库采用采样无标注语料库 *AI Papers* (*S2ORC: The Semantic Scholar Open Research Corpus*, [paper](https://arxiv.org/abs/1911.02782)) 的一部分语料。`--dataset_name`对应`ai_unsup`。

下游任务采样两个计算机领域的数据集，ACL-ARC (*Measuring the Evolution of a Scientific Field through Citation Frames*, [paper](https://aclanthology.org/Q18-1028/))和 SCIERC (*Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction*, [paper](https://arxiv.org/abs/1808.09602v1))，均为分类任务。两个数据集的`--dataset_name`分别对应`acl_sup`和`sci_sup`。还提供对应的few shot数据集，对应`acl_fs`和`sci_fs`。

语料库和数据集可从 [Google Drive link](https://drive.google.com/drive/folders/1xWHB5sXWe7L8I6UENroexo4PAoaFBocX?usp=share_link) 下载并解压，放在代码所在文件夹里。包含语料库`.txt`文件的文件夹、两个分别包含数据集`.jsonl`文件的文件夹一共三个文件夹均和代码同级。

以上数据均由`dataHelper.py`处理，加载成`DatasetDict`形式。

## 扩充词表及tokenizer
首先利用BPE算法从语料库中得到词表，包括`vocab.json`和`merges.txt`：
```
python get_vocab.py
```
把新的token加到原始RoBERTa所用的tokenizer中得到新的tokenizer：
```
python get_new_tokenizer.py
```

## 预训练模型
提供一些预训练模型[Google Drive link](https://drive.google.com/drive/folders/1L76Csml-jStahFMuWWAroE3EuL_FO2TK?usp=share_link)，将在后面说明。

## 训练
本工作使用 Weights & Biases 记录训练过程，如不需要使用，请先删掉命令行中的`--report_to`以及代码`posttraining.py`和`finetuning.py`中的`import wandb`和`wandb.init`。

### Run post training and finetuning
在运行前请先设置好参数。

对于后训练，参数`--use_my_tokenizer`控制是否使用自己的tokenizer，注意同时设置`class ModelArguments`中的参数`--tokenizer_name`指定tokenizer的路径。参数`--use_my_mask`控制是否对于新增token和unk_token设置mask概率为`class DataTrainingArguments`中的`--mlm_probability_large`。进行后训练，运行：
```
python posttraining.py \
    --dataset_name ai_unsup \
    --do_train \
    --use_my_tokenizer False \
    --use_my_mask True \
    --per_device_train_batch_size 128 \
    --max_seq_length 64 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --save_strategy epoch \
    --output_dir ${output_dir} \
    --report_to wandb
```

对于TAPT，运行：
```
python posttraining.py \
	--dataset_name ${dataset_name} \
	--model_name_or_path ${model_name_or_path} \
	--do_train \
	--per_device_train_batch_size 128 \
	--max_seq_length 64 \
	--learning_rate 1e-4 \
	--num_train_epochs 20 \
	--logging_strategy epoch \
	--output_dir ${output_dir} \
	--report_to wandb
```

对于微调，运行：（注：参数`--task_name`和`--use_my_tokenizer`在代码中并没有实际控制作用）
```
python finetuning.py \
  --model_name_or_path ${model} \
  --task_name ${task_name} \
  --dataset_name ${dataset_name} \
  --seed ${seed} \
  --do_train \
  --do_eval \
  --do_predict \
  --use_my_tokenizer True \
  --logging_strategy epoch \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --max_seq_length 164 \
  --per_device_train_batch_size 20 \
  --per_device_eval_batch_size 20 \
  --learning_rate 5e-5 \
  --weight_decay 1e-2 \
  --num_train_epochs 20 \
  --output_dir ${output_dir} \
  --report_to wandb
```

### Run examples
提供了一些后训练和微调的`.sh`文件样例（请先修改其中的预训练模型和输出路径），请在Python代码所在的路径下运行。三种后训练方法分别对应得到三种预训练模型。

Vocab post 为使用扩充词表进行后训练和微调，按正态分布初始化新增tokens的word embeddings（默认）。

Vocab embed 使用扩充词表。对于新增的词，在原有词表上tokenize成sub-words，把这些sub-words对应的 word embeddings取平均作为新增词的初始word embedding，进行后训练。
```
source vocab_embed.sh
```
Mask prob 使用原词表，在后训练中修改mask概率。
```
source mask_prob.sh
```
在两个下游任务上，用五个不同的seed分别运行五次平行微调训练和测试。
```
source tfinetune.sh
```

## 得到最终测试结果
提供代码将不同方法得到的测试结果整理并输出，请先将`write_xlsx.py`中的`pathlst`按照所给样例修改成对应输出路径。也可以在代码中选择输出到表格文件。
```
python write_xlsx.py
```
