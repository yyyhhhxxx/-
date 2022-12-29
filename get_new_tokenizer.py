import logging
import os
import random
import sys
import json
import numpy as np

from transformers import (
    AutoTokenizer,
    RobertaTokenizerFast
)

oripath = 'roberta-base'
addlst = []
cnt = 0

new_tokenizer = RobertaTokenizerFast(vocab_file='my_token3/vocab.json',merges_file='my_token3/merges.txt')

ori_tokenizer = AutoTokenizer.from_pretrained(oripath)
for i in range(len(new_tokenizer)):
    token = new_tokenizer.convert_ids_to_tokens(i)
    Id = ori_tokenizer.convert_tokens_to_ids(token)
    if Id == ori_tokenizer.unk_token_id:
        cnt += 1
        addlst.append(token)

ori_tokenizer.add_tokens(addlst)
path = './new_tokenizer78000/'
os.makedirs(path)
ori_tokenizer.save_pretrained(path)

print(f'New tokenizer saved. Add {cnt} words. Now len(tokenizer)={len(ori_tokenizer)}')