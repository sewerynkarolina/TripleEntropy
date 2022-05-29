import torch
import numpy as np


class TripleEntropyDatasetClassification(torch.utils.data.Dataset):
    def __init__(self, index, df, tokenizer, max_length, sample_size=-1):
        if sample_size != -1:
            index = np.random.choice(index, sample_size, replace=False)

        texts = df.iloc[index]["sentence"].tolist()
        labels = df.iloc[index]["label"].tolist()
        train_encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

        self.encodings = train_encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


class TripleEntropyDatasetNLI(torch.utils.data.Dataset):
    def __init__(self, index, df, tokenizer, max_length, sample_size=-1):
        if sample_size != -1:
            index = np.random.choice(index, sample_size, replace=False)

        labels = df.iloc[index]["label"].tolist()
        questions = df.iloc[index]["question"].tolist()
        sentences = df.iloc[index]["sentence"].tolist()
        texts = list(zip(questions, sentences))
        texts = [txt for txt in texts if isinstance(txt[0], str) and isinstance(txt[1], str)]
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
