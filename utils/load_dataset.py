import pandas as pd
from sklearn.utils import shuffle
from datasets import load_dataset


def prepare_dataset(dataset="sst2", seed=42):
    if dataset == "sst2":
        df = pd.read_csv('data/SST-2/train.tsv', sep='\t')
    elif dataset == "trec":
        dataset = load_dataset('trec')
        df = pd.DataFrame(
            list(zip([(eval['label-coarse']) for eval in dataset['train']],
                     [(eval['text']) for eval in dataset['train']])),
            columns=['label', 'sentence'])
    elif dataset == "mr":
        d = []
        with open('data/MR/rt-polarity.neg', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('data/MR/rt-polarity.pos', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
    elif dataset == "cr":
        d = []
        with open('data/CR/custrev.neg', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('data/CR/custrev.pos', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
    elif dataset == "mpqa":
        d = []
        with open('data/MPQA/mpqa.neg', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('data/MPQA/mpqa.pos', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
    elif dataset == "subj":
        d = []
        with open('data/SUBJ/subj.objective', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('data/SUBJ/subj.subjective', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
    elif dataset == "mrpc":
        df = pd.read_csv('data/MRPC/train.tsv', sep='\t', error_bad_lines=False)
        df = df.rename(columns={'Quality': 'label', '#1 String': 'question', '#2 String': 'sentence'})
        df["question"] = df["question"].astype(str)
        df['sentence'] = df["sentence"].astype(str)
    else:
        raise ValueError(f'Cannot load the dataset: {dataset}.')
    df = shuffle(df, random_state=seed)
    return df
