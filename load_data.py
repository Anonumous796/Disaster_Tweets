from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
import pandas as pd
from utils import clear_text, clean
from transformers import BertTokenizer, BertModel

pandarallel.initialize(progress_bar=True)


class TwitterDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        if self.is_train:
            label = self.data.iloc[idx]['target']
        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                                            padding='max_length', return_attention_mask=True,
                                            return_token_type_ids=True, return_tensors='pt')
        input_ids, attention_mask, token_type_ids = inputs["input_ids"], inputs["attention_mask"], inputs[
            "token_type_ids"]
        if self.is_train:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "label": label
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }


def prepare_dataloaders(model_name: str, max_len=256, batch_size=16):
    train = pd.read_csv('data/train.csv')
    train['text'] = train['text'].parallel_apply(clean)
    # train['text'] = train['text'].parallel_apply(clear_text)
    train, test = train_test_split(train, test_size=0.37, random_state=79)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = TwitterDataset(train, tokenizer, max_len)
    test_dataset = TwitterDataset(test, tokenizer, max_len)
    ll = len(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                  drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                 drop_last=False)
    return train_dataloader, test_dataloader, ll
