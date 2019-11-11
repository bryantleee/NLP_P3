import numpy as np
import torch
import os
from data_loader import fetch_data

# Our stuff we imported
from gensim.models import Word2Vec
import numpy as np
import torch.backends.cudnn as cudnn

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

for i, model in enumerate(models):
    total = 0
    correct = 0
    # ffnn_correct = 0

    for index in range(N):
        input_vector, gold_label = validation_samples[index]
        predicted_vector = model(input_vector)
        predicted_label = torch.argmax(predicted_vector)

        if int(predicted_label == gold_label) == 1:
            correct_outputs[i].append(index)
        else:
            incorrect_outputs[i].append(index)
        correct += int(predicted_label == gold_label)
        total += 1

    print("Validation accuracy for model {}: {}".format(i, correct / total))
    allcorrect = []
    allwrong = []
    for c in correct_outputs[0]:
        if c in correct_outputs[1] and c in correct_outputs[2] and c in correct_outputs[3]:
            allcorrect.append(c)
    for i in incorrect_outputs[0]:
        if i in incorrect_outputs[1] and i in incorrect_outputs[2] and i in incorrect_outputs[3]:
            allwrong.append(i)
    print("Indices of reviews all models get correct: " + str(allcorrect))
    print("length of all correct: " + str(len(allcorrect)))
    print("Indices of reviews all models get wrong: " + str(allwrong))
    print("Length of all wrong: " + str(len(allwrong)))
