import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
from data_loader import fetch_data

# Our stuff we imported
from gensim.models import Word2Vec
from collections import Counter
import numpy as np
import torch.backends.cudnn as cudnn
import adabound

from rnn import RNN
from ffnn import FFNN

# Load models
base = ('rnn_sgd_base.pt', 'ffnn_sgd_base.pt')
hx2 = ('rnn_sgd_hx2.pt', 'ffnn_sgd_hx2.pt')
lx2 = ('rnn_sgd_lx2.pt', 'ffnn_sgd_lx2.pt')

files = (base, hx2, lx2)


base_models = []
#RNN SGD BASE
path = 'models/' + base[0]
model = RNN(32, 1, 64)
model.load_state_dict(torch.load(path))
base_models.append(model)
#FFNN SGD BASE
path = 'models/' + base[1]
model = FFNN(97305, 32, 1)
model.load_state_dict(torch.load(path))
base_models.append(model)

hx2_models = []
#RNN SGD hx2
path = 'models/' + hx2[0]
model = RNN(64, 1, 64)
model.load_state_dict(torch.load(path))
base_models.append(model)
#FFNN SGD hx2
hx2_models = []
path = 'models/' + hx2[1]
model = FFNN(97305, 64, 1)
model.load_state_dict(torch.load(path))
base_models.append(model)


lx2_models = []
#RNN SGD lx2
path = 'models/' + lx2[0]
model = RNN(32, 2, 64)
model.load_state_dict(torch.load(path))
base_models.append(model)
#FFNN SGD lx2
path = 'models/' + lx2[1]
model = FFNN(97305, 32, 2)
model.load_state_dict(torch.load(path))
base_models.append(model)

print('models succesfuly loaded')

# Load trained word embeddings
train_data, valid_data = fetch_data()
wv_model = Word2Vec.load("word2vec.model")

training_samples = []
for t in train_data:
    embedding_list = [wv_model.wv[word] for word in t[0]]
    stacked_embedding = np.stack(embedding_list, axis=0)
    expanded_embedding = np.expand_dims(stacked_embedding, axis=0)
    embedding_tensor = torch.from_numpy(expanded_embedding)
    train_sample = (embedding_tensor, t[1])
    training_samples.append(train_sample)

validation_samples = []
for v in valid_data:
    embedding_list = [wv_model.wv[word] for word in v[0]]
    stacked_embedding = np.stack(embedding_list, axis=0)
    expanded_embedding = np.expand_dims(stacked_embedding, axis=0)
    embedding_tensor = torch.from_numpy(expanded_embedding)
    valid_sample = (embedding_tensor, t[1])
    validation_samples.append(valid_sample)


print('Word embeddings succesfully trained')


# Validate and save models that got it right
both_right = [[],[],[]]
ffnn_right_rnn_wrong = [[],[],[]]
rrn_right_ffnn_wrong = [[],[],[]]

batches = (base_models, hx2_models, lx2_models)

N = len(validation_samples)


for i, batch in enumerate(batches): 
    for index in tqdm(range(N)):
        input_vector, gold_label = validation_samples[index]
        
        predicted_vector_rnn = batch[0](input_vector)
        predicted_label_rnn = torch.argmax(predicted_vector_rnn)

        input_vector, gold_label = valid_data[index] 
        predicted_vector_ffnn = batch[1](input_vector)
        predicted_label_ffnn = torch.argmax(predicted_vector_ffnn)        

        if predicted_label_ffnn == gold_label and predicted_label_rnn == gold_label:
            both_right[i].append(index)
        elif predicted_label_ffnn == gold_label and predicted_label_rnn != gold_label:
            ffnn_right_rnn_wrong[i].append(index)
        elif predicted_label_ffnn != gold_label and predicted_label_rnn == gold_label:
            rrn_right_ffnn_wrong[i].append(index)

        total += 1

  
    output_lists = (both_right, ffnn_right_rnn_wrong, rrn_right_ffnn_wrong)
    
    with open("results.txt","w+") as f:
        for i, correctness_list in enumerate(output_lists):
            for list_of_indices in correctness_list:
                output = ', '.join(list_of_indices)
                print(output)
            
                # f.write()


    # print('Validation accuracy completed for', files[i])
    # print("Validation accuracy for epoch {}: {}".format(10 + 1,  / ))
