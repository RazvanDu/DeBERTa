from transformers import DebertaConfig, DebertaModel, AutoTokenizer
import torch
import numpy as np

context = 512

model = DebertaModel.from_pretrained("microsoft/deberta-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

from datasets import load_dataset

wikipedia = load_dataset("wikipedia", "20220301.en", cache_dir="/home/gigi/hdd/RNNWins/data").with_format(type="torch")

tokens_so_far = 0
crr = 0

dataloader = torch.utils.data.DataLoader(wikipedia['train'], batch_size=1)

sum_tensors = {}
crr_tensors = {}

i = 0



for item in dataloader:

    text = item['text']

    tokenized = tokenizer(text, return_tensors='pt')
    tokens = tokenized['input_ids'][0]

    if len(tokens) <= context:
        continue

    i = np.random.randint(0, len(tokens) - context - 1)

    crr += 1
    tokens_so_far += context

    print(tokens_so_far)

    sliced_output = {key: value[:, i:i+context] for key, value in tokenized.items()}

    outputs = model(**sliced_output)

    for token, emb in zip(tokens, outputs.last_hidden_state[0]):
        if token.item() not in sum_tensors:
            sum_tensors[token.item()] = emb.clone()
            crr_tensors[token.item()] = 1
        else:
            sum_tensors[token.item()] += emb
            crr_tensors[token.item()] += 1

    if tokens_so_far >= 1000000:
        break

print("Finished!")

for token in sum_tensors.keys():
    sum_tensors[token] = sum_tensors[token].clone()/crr_tensors[token]

file_embd = open('deberta_embd.txt', 'w')

file_embd.write(str(len(sum_tensors)) + " " + str(len(sum_tensors[next(iter(sum_tensors))].detach().numpy())) + "\n")

for token in sum_tensors.keys():
    file_embd.write(str(token) + " " + (" ".join([str(x) for x in sum_tensors[token].detach().numpy()])) + "\n")