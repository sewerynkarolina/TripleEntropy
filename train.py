#!/usr/bin/env python
# -*- coding: utf-8 -*-

import transformers
from numpy import unique
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
import training
from RobertaTripleEntropyForSequenceClassification import RobertaTripleEntropyForSequenceClassification
import numpy as np
import pandas as pd
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-ml', '--max-length', default=64, type=int, dest='max_length')
parser.add_argument('--learning-rate', default=1e-5, type=float, dest='learning_rate')
parser.add_argument('--num-warmup-steps', default=10, type=int, dest='num_warmup_steps')
parser.add_argument('--eps', default=1e-08, type=float, dest='eps')
parser.add_argument('--model-name', default="roberta-base", dest='model_name')
parser.add_argument('--weight-decay', default=0.01, type=float, dest='weight_decay')
parser.add_argument('--la', default=3.3, type=float, dest='la')
parser.add_argument('--gamma', default=0.1, type=float, dest='gamma')
parser.add_argument('--margin', default=0.1, type=float, dest='margin')
parser.add_argument('--centers', default=1000, type=int, dest='centers')
parser.add_argument('--beta', default=0.9, type=float, dest='beta')
parser.add_argument('--seed', default=2048, type=int, dest='seed')
parser.add_argument('--output-dir', default="./result", dest='output_dir')
parser.add_argument('--epochs', default=8, type=int, dest='epochs')
parser.add_argument('--num-training-steps', default=10320, type=int, dest='num_training_steps')


args = parser.parse_args()

TrainingArguments(
    output_dir=args.output_dir,  # output directory
    num_train_epochs=args.epochs
)

if __name__ == '__main__':
    seed = args.seed
    max_length = args.max_length
    learning_rate = args.learning_rate
    num_warmup_steps = args.num_warmup_steps
    num_training_steps = args.num_training_steps
    eps = args.eps
    model_name = args.model_name
    weight_decay = args.weight_decay
    la = args.la
    gamma = args.gamma
    margin = args.margin
    centers = args.centers
    beta = args.beta

    np.random.seed(seed)
    training.set_seed(seed)

    dataset = load_dataset('trec')
    df = pd.DataFrame(
        list(zip([(eval['label-coarse']) for eval in dataset['train']],
                 [(eval['text']) for eval in dataset['train']])),
        columns=['label', 'sentence'])
    df_valid = pd.DataFrame(
        list(zip([(eval['label-coarse']) for eval in dataset['test']],
                 [(eval['text']) for eval in dataset['test']])),
        columns=['label', 'sentence'])

    train_texts = df["sentence"].tolist()
    train_labels = df["label"].tolist()

    valid_texts = df_valid["sentence"].tolist()
    valid_labels = df_valid["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

    train_dataset = training.B2CContractClausesDataset(train_encodings, train_labels)
    valid_dataset = training.B2CContractClausesDataset(valid_encodings, valid_labels)

    model = RobertaTripleEntropyForSequenceClassification.from_pretrained(model_name,
                                                                          num_labels=len(unique(df.label)), la=la,
                                                                          gamma=gamma,
                                                                          margin=margin, centers=centers, beta=beta)

    param_groups = [{"params": model.parameters(),
                     'lr': float(learning_rate)}]

    optimizer = transformers.AdamW(param_groups, eps=eps, weight_decay=weight_decay, correct_bias=True)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=num_warmup_steps,
                                                             num_training_steps=num_training_steps)
    optimizers = optimizer, scheduler
    training.set_seed(seed)

    trainer = Trainer(
        model=model,
        args=training.training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        optimizers=optimizers
    )

    trainer.train()
    trainer.evaluate()
