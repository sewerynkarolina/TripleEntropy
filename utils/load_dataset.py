import pandas as pd
import numpy as np

def load_dataset(dataset="sst2", size=20, seed=42):
    if dataset == "sst2":
        df = pd.read_csv('SST-2/train.tsv', sep='\t')
        df_valid = pd.read_csv('SST-2/dev.tsv', sep='\t')
        type_ds = "SST2"
    elif dataset == "trec":
        dataset = load_dataset('trec')
        df = pd.DataFrame(
            list(zip([(eval['label-coarse']) for eval in dataset['train']],
                     [(eval['text']) for eval in dataset['train']])),
            columns=['label', 'sentence'])
        df_valid = pd.DataFrame(
            list(zip([(eval['label-coarse']) for eval in dataset['test']],
                     [(eval['text']) for eval in dataset['test']])),
            columns=['label', 'sentence'])
    elif dataset == "mr":
        d = []
        with open('MR/rt-polarity.neg', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('MR/rt-polarity.pos', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
        from sklearn.utils import shuffle

        df = shuffle(df, random_state=seed)
        type_ds = "MR"

    elif dataset == "cr":
        d = []
        with open('CR/custrev.neg', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('CR/custrev.pos', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
        from sklearn.utils import shuffle

        df = shuffle(df, random_state=seed)
        type_ds = "CR"
    elif dataset == "mpqa":
        d = []
        with open('MPQA/mpqa.neg', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('MPQA/mpqa.pos', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
        from sklearn.utils import shuffle

        df = shuffle(df, random_state=seed)
        df = shuffle(df, random_state=seed)
        type_ds = "MPQA"

    elif dataset =="subj":
        d = []
        with open('SUBJ/subj.objective', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 0
                    }
                )
        with open('SUBJ/subj.subjective', "r") as f:
            for elem in f.readlines():
                d.append(
                    {
                        'sentence': elem,
                        'label': 1
                    }
                )
        df = pd.DataFrame(d)
        from sklearn.utils import shuffle

        df = shuffle(df, random_state=seed)
        type_ds = "SUBJ"

    train_index = np.random.choice(list(range(df.shape[0])), size, replace=False)
