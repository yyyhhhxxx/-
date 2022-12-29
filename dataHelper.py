import json
import jsonlines
import random
import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_dataset_builder

random.seed(2022)
acl_label_map = {"CompareOrContrast":0, "Background":1, "Motivation":2, "Uses":3, "Future":4, "Extends":5}
sci_label_map = {'CONJUNCTION': 0, 'FEATURE-OF': 1, 'HYPONYM-OF': 2, 'USED-FOR': 3, 'PART-OF': 4, 'COMPARE': 5, 'EVALUATE-FOR': 6}

def sample_8(cls_len_):
	ans = [random.sample(x, 8) for x in cls_len_]
	return ans

def sample_from_dataset(ds, num_cls):
	'''
	ds: Dataset()
	'''
	ans = {"text":[], "labels":[]}
	full_text = ds["text"]
	full_labels = ds["labels"]
	unique_labels = ds.unique("labels")
	cls_text_lst = {x: [] for x in unique_labels}
	cls_fs_text_lst = []
	for text_, label_ in zip(full_text, full_labels):
		cls_text_lst[label_].append(text_)


	cls_len = [[i for i in range(len(x))] for x in cls_text_lst.values()]
	cls_fs_idx = sample_8(cls_len)
	for o in range(num_cls):
		cls_fs_text_lst += [cls_text_lst[unique_labels[o]][i] for i in cls_fs_idx[o]]
		ans["labels"] += [unique_labels[o] for i in range(8)]

	ans["text"] = cls_fs_text_lst

	return Dataset.from_dict(ans)

def get_acl_unsup():
    data_files = {'train': './acl_unsup/acl_anthology.txt'}
    datasets = load_dataset('text', data_files=data_files)

    return datasets


def get_ai_unsup():
    data_files = {'train': './ai_unsup/ai_corpus.txt'}
    datasets = load_dataset('text', data_files=data_files)

    return datasets

def get_dataset(dataset_name):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''

	if dataset_name == 'ai_unsup':
		datasets = get_ai_unsup()
		return datasets

	if dataset_name == 'acl_unsup':
		datasets = get_acl_unsup()
		return datasets

	if dataset_name == 'unsup':
		ds_ai = get_ai_unsup()
		ds_acl = get_acl_unsup()
		text_lst = ds_ai['train']['text'] + ds_acl['train']['text']
		datasets = DatasetDict({'train': Dataset.from_dict({'text': text_lst})})
		return datasets

	dataset = DatasetDict()
	train_dict = {"text":[], "labels":[]}
	test_dict = {"text":[], "labels":[]}
	dev_dict = {"text":[], "labels":[]} # validation

	if dataset_name.find('acl') != -1:
		with jsonlines.open('aclarc/test.jsonl') as reader:
			for obj in reader:
				test_dict["text"].append(obj["text"])
				test_dict["labels"].append(acl_label_map[obj["label"]])

		with jsonlines.open('aclarc/dev.jsonl') as reader1:
			for obj in reader1:
				dev_dict["text"].append(obj["text"])
				dev_dict["labels"].append(acl_label_map[obj["label"]])

		with jsonlines.open('aclarc/train.jsonl') as reader2:
			for obj in reader2:
				train_dict["text"].append(obj["text"])
				train_dict["labels"].append(acl_label_map[obj["label"]])				

	elif dataset_name.find('sci') != -1:
		with jsonlines.open('scierc/test.jsonl') as reader:
			for obj in reader:
				test_dict["text"].append(obj["text"])
				test_dict["labels"].append(sci_label_map[obj["label"]])	

		with jsonlines.open('scierc/dev.jsonl') as reader1:
			for obj in reader1:
				dev_dict["text"].append(obj["text"])
				dev_dict["labels"].append(sci_label_map[obj["label"]])

		with jsonlines.open('scierc/train.jsonl') as reader2:
			for obj in reader2:
				train_dict["text"].append(obj["text"])
				train_dict["labels"].append(sci_label_map[obj["label"]])

	dataset["train"] = Dataset.from_dict(train_dict)
	dataset["test"] = Dataset.from_dict(test_dict)
	dataset["dev"] = Dataset.from_dict(dev_dict)

	if dataset_name[-3:] != 'sup':
		train_dataset = sample_from_dataset(dataset["train"], len(dataset["train"].unique("labels")))
		dataset["train"] = train_dataset

	return dataset
