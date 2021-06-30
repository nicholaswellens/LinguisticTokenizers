import torch
import numpy as np
from numpy import array, argmax
from transformers import BertTokenizer, BertForMaskedLM, AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback

import random
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

#from keras.utils import to_categorical
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from collections import defaultdict
import pickle


#########################################
########## VARIABLES ####################
#########################################
print("Prepare Variables")

MAX_LEN = 128
#BATCH_SIZE = 128
#EPOCHS = 50
lr = 1e-4
RANDOM_SEED = 56
VOCAB_SIZE = 30522

vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/EUROPARLvocabFROGMORPHHASHTAG.txt"

print("Prepare Seeds")
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



#########################################
######### PREPARE DATA ##################
#########################################
print("Prepare Europarl Data")

with open('TrainFiles/FrogPieceHashtag_europarl_sentences_ids_1000k_128seq.pickle', 'rb') as inputfile:
    europarl_sentences_ids_1000k_128seq = pickle.load(inputfile)

with open('TrainFiles/FrogPieceHashtag_europarl_sentences_ids_1000k_2000k_128seq.pickle', 'rb') as inputfile:
    europarl_sentences_ids_1000k_2000k_128seq = pickle.load(inputfile)

Europarl_sentences_ids = europarl_sentences_ids_1000k_128seq + europarl_sentences_ids_1000k_2000k_128seq
print(Europarl_sentences_ids[:10])

print("Prepare Oscar Data")

with open('TrainFiles/FrogPieceHashtag_oscar_sentences_ids_500k_128seq.pickle', 'rb') as inputfile:
    oscar_sentences_ids_500k_128seq = pickle.load(inputfile)

with open('TrainFiles/FrogPieceHashtag_oscar_sentences_ids_500k_1000k_128seq.pickle', 'rb') as inputfile:
    oscar_sentences_ids_500k_1000k_128seq = pickle.load(inputfile)

with open('TrainFiles/FrogPieceHashtag_oscar_sentences_ids_1000k_1500k_128seq.pickle', 'rb') as inputfile:
    oscar_sentences_ids_1000k_1500k_128seq = pickle.load(inputfile)

with open('TrainFiles/FrogPieceHashtag_oscar_sentences_ids_1500k_2000k_128seq.pickle', 'rb') as inputfile:
    oscar_sentences_ids_1500k_2000k_128seq = pickle.load(inputfile)

print(europarl_sentences_ids_1000k_128seq[:2])

Oscar_sentences_ids = oscar_sentences_ids_500k_128seq + oscar_sentences_ids_500k_1000k_128seq + oscar_sentences_ids_1000k_1500k_128seq + oscar_sentences_ids_1500k_2000k_128seq



train_europarl_data, eval_europarl_data = train_test_split(Europarl_sentences_ids, test_size=0.05)
train_oscar_data, eval_oscar_data = train_test_split(Oscar_sentences_ids, test_size=0.05)

pretraining_data = train_europarl_data + train_oscar_data
eval_data = eval_europarl_data + eval_oscar_data

print(len(Europarl_sentences_ids))
print(len(Oscar_sentences_ids))

print(len(train_europarl_data))
print(len(eval_europarl_data))
print(len(train_oscar_data))
print(len(eval_oscar_data))

print(len(pretraining_data))
print(len(eval_data))


pretraining_data = np.array(pretraining_data)
eval_data = np.array(eval_data)

#train_test_split() 5% and 5%



class BertTinyDataset(Dataset):

    def __init__(self, input_sentences_ids, max_len, vocab_size):
        self.inputs = input_sentences_ids
        self.max_len = max_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        input_ids = self.inputs[item]
        input_ids = torch.tensor(input_ids)

        return input_ids


#########################################
######### PREPARE TOKENIZER #############
#########################################
print("Prepare Data_collator Tokenizer")

MorphTokenizer_for_data_collator = BertTokenizer(vocab_file=vocabulary_file_to_use_in_transformers, do_lower_case=True, do_basic_tokenize=True, padding_side="right",tokenize_chinese_chars=False)

#########################################
######### PREPARE MODEL #################
#########################################
print("Prepare Model")

config = AutoConfig.from_pretrained('prajjwal1/bert-tiny')
model = BertForMaskedLM(config)
print(f"This model has {model.num_parameters()} parameters")


Train_Dataset = BertTinyDataset(pretraining_data, MAX_LEN, VOCAB_SIZE)
Eval_Dataset = BertTinyDataset(eval_data, MAX_LEN, VOCAB_SIZE)

print("Prepare Data Collator")
# Need to add, as the default_collator does not do mlm preperation
data_collator = DataCollatorForLanguageModeling(
    tokenizer = MorphTokenizer_for_data_collator, mlm=True, mlm_probability=0.15
)

print("Prepare Training Arguments")
training_args = TrainingArguments(
    output_dir="./output_dir_hashtag",
    overwrite_output_dir=True,
    num_train_epochs=60,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=5000,
    save_total_limit=3,
    weight_decay=0.01,
    learning_rate=0.0001,
    prediction_loss_only=True,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    seed=RANDOM_SEED
    #eval_steps = 300000
)

print("Prepare Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=Train_Dataset,
    eval_dataset=Eval_Dataset,
    callbacks=[EarlyStoppingCallback(2,0)]
)


print("Starting Training")
trainer.train()

print("Saving trained model")
trainer.save_model("./FrogPieceHashtag128")
