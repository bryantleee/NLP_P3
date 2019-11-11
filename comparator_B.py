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

from rnn import RNN
# from ffnn import FFNN, convert_to_vector_representation, make_indices, make_vocab

directory = 'models_b/'
model_paths = ['rnn_sgd_base.pt', 'rnn_rmsprop_base.pt', 'lstm_sgd_base.pt', 'lstm_rmsprop_base.pt']
model1 = RNN(32, 1, 64, True)
model1.load_state_dict(torch.load(os.path.join(directory, model_paths[0])))
model2 = RNN(32, 1, 64, True)
model2.load_state_dict(torch.load(os.path.join(directory, model_paths[1])))
model3 = RNN(32, 1, 64, False)
model3.load_state_dict(torch.load(os.path.join(directory, model_paths[2])))
model4 = RNN(32, 1, 64, False)
model4.load_state_dict(torch.load(os.path.join(directory, model_paths[3])))
models = [model1, model2, model3, model4]

print('models succesfuly loaded')

# Load trained word embeddings
train_data, valid_data = fetch_data()
wv_model = Word2Vec.load("word2vec.model")

validation_samples = []
for v in valid_data:
    embedding_list = [wv_model.wv[word] for word in v[0]]
    stacked_embedding = np.stack(embedding_list, axis=0)
    expanded_embedding = np.expand_dims(stacked_embedding, axis=0)
    embedding_tensor = torch.from_numpy(expanded_embedding)
    valid_sample = (embedding_tensor, v[1])
    validation_samples.append(valid_sample)

N = len(validation_samples)

print('starting validation counts')

correct_outputs = [[], [], [], []]
incorrect_outputs = [[], [], [], []]

rating_sums = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
rating_totals = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

for i, model in enumerate(models):
    total = 0
    correct = 0

    for index in range(N):
        input_vector, gold_label = validation_samples[index]
        predicted_vector = model(input_vector)
        predicted_label = torch.argmax(predicted_vector)
        rating_totals[i][gold_label] += 1
        if int(predicted_label == gold_label) == 1:
            rating_sums[i][gold_label] += 1
            correct_outputs[i].append(index)
        else:
            incorrect_outputs[i].append(index)
        correct += int(predicted_label == gold_label)
        total += 1

    print("Validation accuracy for model {}: {}".format(i, correct / total))

for i in range(4):
    for j in range(5):
        print('Model {} rating {} percent correct: {}'.format(i, j, rating_sums[i][j]/rating_totals[i][j]))
