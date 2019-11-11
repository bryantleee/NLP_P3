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
from collections import Counter

from rnn import RNN
# from ffnn import FFNN, convert_to_vector_representation, make_indices, make_vocab

directory = 'models_part_b/'
model_paths = ['rnn_sgd_lr-10.pt', 'rnn_rmsprop_lr-10.pt', 'lstm_sgd_lr-10.pt', 'lstm_rmsprop_lr-10.pt']
model1 = RNN(32, 1, 64, True)
model1.load_state_dict(torch.load(os.path.join(directory, model_paths[0])))
model2 = RNN(32, 1, 64, True)
model2.load_state_dict(torch.load(os.path.join(directory, model_paths[1])))
model3 = RNN(32, 1, 64, False)
model3.load_state_dict(torch.load(os.path.join(directory, model_paths[2])))
model4 = RNN(32, 1, 64, False)
model4.load_state_dict(torch.load(os.path.join(directory, model_paths[3])))
models = [model1, model2, model3, model4]

# models = [RNN(32, 1, 64, True).load_state_dict(torch.load(os.path.join(directory, model_paths[0]))),
#     RNN(32, 1, 64, True).load_state_dict(torch.load(os.path.join(directory, model_paths[1]))),
#     RNN(32, 1, 64, False).load_state_dict(torch.load(os.path.join(directory, model_paths[2]))),
#     RNN(32, 1, 64, False).load_state_dict(torch.load(os.path.join(directory, model_paths[3])))]

#FFNN SGD BASE
# path = directory + base[1]
# model = FFNN(97305, 32, 1)
# model.load_state_dict(torch.load(path))
# base_models.append(model)



# Load models
# base = ('rnn_sgd_base.pt', 'ffnn_sgd_base.pt')
# hx2 = ('rnn_sgd_hx2.pt', 'ffnn_sgd_hx2.pt')
# lx2 = ('rnn_sgd_lx2.pt', 'ffnn_sgd_lx2.pt')
#
# files = (base, hx2, lx2)
# directory = 'models_part_a/'
# base_models = []
# #RNN SGD BASE
# path = directory + base[0]
# model = RNN(32, 1, 64)
# model.load_state_dict(torch.load(path))
# base_models.append(model)
# #FFNN SGD BASE
# path = directory + base[1]
# model = FFNN(97305, 32, 1)
# model.load_state_dict(torch.load(path))
# base_models.append(model)
#
# hx2_models = []
# #RNN SGD hx2
# path = directory + hx2[0]
# model = RNN(64, 1, 64)
# model.load_state_dict(torch.load(path))
# hx2_models.append(model)
# #FFNN SGD hx2
# path = directory + hx2[1]
# model = FFNN(97305, 64, 1)
# model.load_state_dict(torch.load(path))
# hx2_models.append(model)
#
#
# lx2_models = []
# #RNN SGD lx2
# path = directory + lx2[0]
# model = RNN(32, 2, 64)
# model.load_state_dict(torch.load(path))
# lx2_models.append(model)
# #FFNN SGD lx2
# path = directory + lx2[1]
# model = FFNN(97305, 32, 2)
# model.load_state_dict(torch.load(path))
# lx2_models.append(model)

print('models succesfuly loaded')

# Load trained word embeddings
train_data, valid_data = fetch_data()
wv_model = Word2Vec.load("word2vec.model")

# training_samples = []
# for t in train_data:
#     embedding_list = [wv_model.wv[word] for word in t[0]]
#     stacked_embedding = np.stack(embedding_list, axis=0)
#     expanded_embedding = np.expand_dims(stacked_embedding, axis=0)
#     embedding_tensor = torch.from_numpy(expanded_embedding)
#     train_sample = (embedding_tensor, t[1])
#     training_samples.append(train_sample)

validation_samples = []
for v in valid_data:
    embedding_list = [wv_model.wv[word] for word in v[0]]
    stacked_embedding = np.stack(embedding_list, axis=0)
    expanded_embedding = np.expand_dims(stacked_embedding, axis=0)
    embedding_tensor = torch.from_numpy(expanded_embedding)
    valid_sample = (embedding_tensor, v[1])
    validation_samples.append(valid_sample)


# print('Word embeddings succesfully trained')
# vocab = make_vocab(train_data)
# vocab, word2index, index2word = make_indices(vocab)
#
# train_data = convert_to_vector_representation(train_data, word2index)
# valid_data = convert_to_vector_representation(valid_data, word2index)
# print('word embeddings succesfully converted to vectors')


# Validate and save models that got it right
# both_right = [[],[],[]]
# ffnn_right_rnn_wrong = [[],[],[]]
# rnn_right_ffnn_wrong = [[],[],[]]
#
# batches = (base_models, hx2_models, lx2_models)

N = len(validation_samples)

# review_counter = Counter()

print('starting validation counts')

# rnn_average_dist = [[], [], []]
# ffnn_average_dist = [[], [], []]

correct_outputs = [[], [], [], []]
incorrect_outputs = [[], [], [], []]

rating_sums = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
rating_totals = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

for i, model in enumerate(models):
    total = 0
    correct = 0
    # ffnn_correct = 0

    for index in range(N):
        input_vector, gold_label = validation_samples[index]
        predicted_vector = model(input_vector)
        predicted_label = torch.argmax(predicted_vector)

        # input_vector, gold_label = valid_data[index]

        # predicted_vector_ffnn = batch[1](input_vector)
        # predicted_label_ffnn = torch.argmax(predicted_vector_ffnn)

        # review_counter.update(gold_label)

        # rnn_average_dist[i] += abs(gold_label - predicted_label_rnn.item())
        # ffnn_average_dist[i] += abs(gold_label - predicted_label_ffnn.item())

        # if predicted_label_ffnn == gold_label and predicted_label_rnn == gold_label:
        #     both_right[i].append(index)
        #     rnn_correct += 1
        #     ffnn_correct += 1

        # elif predicted_label_ffnn == gold_label and predicted_label_rnn != gold_label:
        #     ffnn_right_rnn_wrong[i].append(index)
        #     ffnn_correct += 1
        #
        # elif predicted_label_ffnn != gold_label and predicted_label_rnn == gold_label:
        #     rnn_right_ffnn_wrong[i].append(index)
        #     rnn_correct += 1
        rating_totals[i][gold_label] += 1
        if int(predicted_label == gold_label) == 1:
            rating_sums[i][gold_label] += 1
            correct_outputs[i].append(index)
        else:
            incorrect_outputs[i].append(index)
        correct += int(predicted_label == gold_label)
        total += 1

    print("Validation accuracy for model {}: {}".format(i, correct / total))
    # print("Validation accuracy for ffnn, batch {}: {}".format(i , correct / total))

for i in range(4):
    for j in range(5):
        print('Model {} rating {} percent correct: {}'.format(i, j, rating_sums[i][j]/rating_totals[i][j]))
# rnn_average_distances = [average_sum/N for average_sum in rnn_average_dist]
# ffnn_average_distances = [average_sum/N for average_sum in ffnn_average_dist]
#
# print('RNN AVG DISTANCES:', rnn_average_distances)
# print('FFNN AVG DISTANCES:', ffnn_average_distances)
#
#
# for i, average_sum in enumerate(rnn_average_dist):
#     rnn_average_dist[i] = average_sum/N
#
# for i, batch in enumerate(both_right):
#     both_right[i] = list(map(str, both_right[i]))
# for i, batch in enumerate(both_right):
#     ffnn_right_rnn_wrong[i] = list(map(str, ffnn_right_rnn_wrong[i]))
# for i, batch in enumerate(both_right):
#     rnn_right_ffnn_wrong[i] = list(map(str, rnn_right_ffnn_wrong[i]))
#
# with open("comparator_outputs/base.txt","w+") as f:
#     f.write('both correct: \n\n')
#     f.write( ", ".join(both_right[0]))
#
#     f.write('\nRNN correct: \n\n')
#     f.write( ", ".join(rnn_right_ffnn_wrong[0]))
#
#     f.write('\nFFNN correct: \n\n')
#     f.write( ", ".join(ffnn_right_rnn_wrong[0]))
#
#
# with open("comparator_outputs/hx2.txt","w+") as f:
#     f.write('both correct: \n\n')
#     f.write( ", ".join(both_right[1]))
#
#     f.write('\nRNN correct: \n\n')
#     f.write( ", ".join(rnn_right_ffnn_wrong[1]))
#
#     f.write('\nFFNN correct: \n\n')
#     f.write( ", ".join(ffnn_right_rnn_wrong[1]))
#
# with open("comparator_outputs/lx2.txt","w+") as f:
#     f.write('both correct: \n\n')
#     f.write( ", ".join(both_right[2]))
#
#     f.write('\nRNN correct: \n\n')
#     f.write( ", ".join(rnn_right_ffnn_wrong[2]))
#
#     f.write('\nFFNN correct: \n\n')
#     f.write( ", ".join(ffnn_right_rnn_wrong[2]))
