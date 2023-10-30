from transformers import DebertaConfig, DebertaModel, AutoTokenizer
import torch

model = DebertaModel.from_pretrained("microsoft/deberta-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

from datasets import load_dataset

wikipedia = load_dataset("wikipedia", "20220301.en", cache_dir="/home/gigi/hdd/RNNWins/data").with_format(type="torch")

tokens_so_far = 0

dataloader = torch.utils.data.DataLoader(wikipedia['train'], batch_size=1)

for item in dataloader:

    text = item['text']
    print(text)

    tokens_so_far += len(tokenizer(text)['input_ids'])

    if tokens_so_far > 10000:
        break



from transformers import DebertaTokenizer

inputs = tokenizer(" cute my dog is cute", return_tensors="pt")

print(tokenizer.batch_decode(inputs['input_ids']))
print(tokenizer.decode(inputs['input_ids'][0][6]))

print(inputs['input_ids'][0])

outputs = model(**inputs)

#print(outputs.last_hidden_state[0][0])