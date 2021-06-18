# 0 비추천 1 추천

import jsonlines
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
import transformers
print(transformers.__version__)
import mxnet
print(mxnet.__version__)
from transformers.optimization import get_cosine_schedule_with_warmup


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        output = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                           attention_mask=attention_mask.float().to(token_ids.device))

        if self.dr_rate:
            out = self.dropout(output[1])
        return self.classifier(out)


# GPU 사용 시
device = torch.device("cuda:0")
print(device)

model = torch.load('./capdi_bert.pt', map_location='cpu').to(device)
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
print("load model")

# ============================================================================
# Setting parameters
max_len = 64
batch_size = 16
warmup_ratio = 0.1
num_epochs = 3
max_grad_norm = 1
log_interval = 100
learning_rate = 5e-5


# ============================================================================
dataset_test = []
pd.set_option('display.max_columns', None)
df = pd.read_csv('./test2.csv')
print(df)

df['review'] = df['review'].replace(to_replace=np.nan, value='.')
for i in df['review']:
    dataset_test.append(i)
print(dataset_test)

nan_list_index = []
for e in range(len(df['review'])):
    if df['review'][e] == '.':
        nan_list_index.append(e)
print("nan_list_index", len(nan_list_index), nan_list_index)


# ============================================================================
class BERTDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i]

    def __len__(self):
        return len(self.sentences)


data_test = BERTDataset(dataset_test, tok, max_len, True, False)

test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)
print("create test dataset")

for e, i in enumerate(test_dataloader):
    if e < 5:
        print(i)
    else:
        break
# ============================================================================
print("Inference")
predict = []
model.eval()
with torch.no_grad():
    for batch_id, (token_ids, valid_length, segment_ids) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        out = model(token_ids, valid_length, segment_ids)

        predict.extend(torch.argmax(out, dim=-1).data.cpu().numpy())

predict = np.array(predict)

# nan 값 score 참고해서 예외 처리
for i in nan_list_index:
    if int(df['score'][i]) > 5:
        predict[i] = 1
    else:
        predict[i] = 0

# ============================================================================
df['predict'] = predict
df.to_csv('inference.csv', index=False)
