from transformers import DebertaConfig, DebertaModel, AutoTokenizer
import torch
import numpy as np

device = "cuda"
torch.set_default_device(device)

context = 512

model = DebertaModel.from_pretrained("microsoft/deberta-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

print(len(tokenizer.get_vocab()))

from datasets import load_dataset

wikipedia = load_dataset("wikipedia", "20220301.en", cache_dir="/home/gigi/hdd/RNNWins/data").with_format(type="torch")

tokens_so_far = 0
crr = 0

dataloader = torch.utils.data.DataLoader(wikipedia['train'], batch_size=1)

sum_tensors = {}
crr_tensors = {}

i = 0

print(len(dataloader))

for item in dataloader:

    text = item['text']

    tokenized = tokenizer(text, return_tensors='pt')

    #print(tokenizer.vocab['the'])
    #print(text)
    #for i in range(50):
    #    token_id = tokenized['input_ids'][0][i].item()
    #    token_str = tokenizer.deco[token_id]
    #    vocab_id = tokenizer.vocab[token_str]
    #    print(token_id, ' ', token_str, ' ', vocab_id)
    #break

    tokens = tokenized['input_ids'][0]

    if len(tokens) <= context + 1:
        continue

    i = np.random.randint(0, len(tokens) - context - 1)

    crr += 1
    tokens_so_far += context

    print(tokens_so_far)

    sliced_output = {key: value[:, i:i+context] for key, value in tokenized.items()}

    outputs = model(**sliced_output)

    for token, emb in zip(tokens, outputs.last_hidden_state[0]):
        if token.item() not in sum_tensors:
            sum_tensors[token.item()] = emb.detach()
            crr_tensors[token.item()] = 1
        else:
            sum_tensors[token.item()] += emb.detach()
            crr_tensors[token.item()] += 1

    if tokens_so_far >= 100000000: # 10 million
        break

print("Finished!")

number_seen_once = 0
for value in crr_tensors.values():
    if value == 1:
        number_seen_once += 1

print(str(number_seen_once) + " tokens have only been seen once!")

unseen_tokens = []
crr = 0

for unseen in tokenizer.get_vocab():
    crr += 1
    if crr%1000 == 0:
        print(crr)
    #print("TEMP ", tokenizer.vocab[unseen], " ", unseen)
    if tokenizer.vocab[unseen] not in sum_tensors.keys():
        unseen_tokens.append(unseen)

print("Number of tokens unseen: ", (len(tokenizer.get_vocab())-len(sum_tensors)), '/', len(tokenizer.get_vocab()))

print("Unseen tokens list: ", unseen_tokens)
print("Number of unseen tokens metric 2: ", len(unseen_tokens))

for token in sum_tensors.keys():
    sum_tensors[token] = sum_tensors[token].clone()/crr_tensors[token]

file_embd = open('deberta_embd.tsv', 'w', encoding="utf-8")

file_embd.write(str(len(sum_tensors)) + " " + str(len(sum_tensors[next(iter(sum_tensors))].cpu().detach().numpy())) + "\n")
for token in sum_tensors.keys():
    file_embd.write(str(tokenizer.convert_ids_to_tokens([token])[0]) + " " + (" ".join([str(x) for x in sum_tensors[token].cpu().detach().numpy()])) + "\n")
    #print("TEST ", , " ", tokenizer.decode(token))