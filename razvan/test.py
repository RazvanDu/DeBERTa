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

i = 0

for item in dataloader:

    text = item['text']
    #print(text)

    tokenized = tokenizer(text, return_tensors='pt')
    tokens = tokenized['input_ids'][0]

    if len(tokens) <= context:
        continue

    i = np.random.randint(0, len(tokens) - context - 1)

    crr += 1

    print(crr)

    if crr > 10:
        break

    sliced_output = {key: value[:, i:i+context] for key, value in tokenized.items()}

    #print(tokenized)
    #print(sliced_output)

    outputs = model(**sliced_output)

    print(len(outputs.last_hidden_state[0]))


from transformers import DebertaTokenizer

inputs = tokenizer(" cute my dog is cute", return_tensors="pt")

print(tokenizer.batch_decode(inputs['input_ids']))
print(tokenizer.decode(inputs['input_ids'][0][6]))

print(inputs['input_ids'][0])

outputs = model(**inputs)

#print(outputs.last_hidden_state[0][0])