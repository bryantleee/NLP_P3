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


unk = '<UNK>'


# New base settings: hidden=32, layers=1, epochs=10, embedding=64
# NOTE: adam, lstm on small network, adabound,


class RNN(nn.Module):
    def __init__(self, hidden_dim, n_layers, embedding_dim):  # Add relevant parameters
        super(RNN, self).__init__()
        # Fill in relevant parameters
        # Ensure parameters are initialized to small values, see PyTorch documentation for guidance

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.Linear = nn.Linear(hidden_dim, 5)
        self.softmax = nn.LogSoftmax(dim=0)
        self.criterion = nn.NLLLoss()

        # self.W = nn.Linear(embedding_dim, hidden_dim)
        # self.U = nn.Linear(hidden_dim, hidden_dim)
        # self.V = nn.Linear(hidden_dim, 5)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.criterion(predicted_vector, gold_label)

    def forward(self, inputs):
        # h = torch.zeros(self.hidden_dim)
        # for input_vector in inputs:
        #     h = self.W(input_vector) + self.U(h)
        # output = self.V(h)
        # predicted_vector = self.softmax(output)

        # begin code
        batch_size = inputs.size()[0]
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        # output, h_n = self.rnn(inputs, h_0)
        c_0 = h_0.clone()
        output, h_n = self.lstm(inputs, (h_0, c_0))

        distribution = self.Linear(output[0][-1])
        predicted_vector = self.softmax(distribution)
        # out = self.Linear(torch.squeeze(h_n))
        # predicted_vector = self.softmax(out)
        # Remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
        # end code
        return predicted_vector

# You may find the functions make_vocab() and make_indices from ffnn.py useful; you are free to copy them directly (or call those functions from this file)


def main(name, embedding_dim, hidden_dim, n_layers, epochs):  # Add relevant parameters
    # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    train_data, valid_data = fetch_data()

    # counts = Counter()
    # review_list = []
    # for t in train_data:
    #     review_list.append(t[0])
    #     counts.update(t[0])
    # for v in valid_data:
    #     review_list.append(v[0])
    #     counts.update(v[0])
    # model = Word2Vec(review_list, size=embedding_dim, min_count=1)
    # model.save('word2vec.model')

    model = Word2Vec.load("word2vec.model")

    # create list of word embeddings for each review
    # reshape training data and validation data embeddings and tensorify
    training_samples = []
    for t in train_data:
        embedding_list = [model.wv[word] for word in t[0]]
        stacked_embedding = np.stack(embedding_list, axis=0)
        expanded_embedding = np.expand_dims(stacked_embedding, axis=0)
        embedding_tensor = torch.from_numpy(expanded_embedding)
        # embedding_tensor = torch.from_numpy(stacked_embedding)
        train_sample = (embedding_tensor, t[1])
        training_samples.append(train_sample)

    validation_samples = []
    for v in valid_data:
        embedding_list = [model.wv[word] for word in v[0]]
        stacked_embedding = np.stack(embedding_list, axis=0)
        expanded_embedding = np.expand_dims(stacked_embedding, axis=0)
        embedding_tensor = torch.from_numpy(expanded_embedding)
        # embedding_tensor = torch.from_numpy(stacked_embedding)
        valid_sample = (embedding_tensor, t[1])
        validation_samples.append(valid_sample)
    # TODO: need to reshape embeddings for validation data too (reshaped_training must be built in the same way as reshaped_training)
    # TODO: also reshaped embeddings must become tuples with y labels attached to be able to shuffle later
    # so reshaped_training[0] = (tensor{document}, Ylabel)

    model = RNN(hidden_dim, n_layers, embedding_dim)  # Fill in parameters
    # print(model(reshaped_training[0]))

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.3e-3)
    # optimizer = adabound.AdaBound(model.parameters(), lr=3e-4, final_lr=3e-3)
    # optimizer = optim.RMSprop(model.parameters(), lr=0.3e-3)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(training_samples) # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(training_samples)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = training_samples[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(validation_samples) # Good practice to shuffle order of validation data
        # minibatch_size = 16
        # for minibatch_index in tqdm(range(N // minibatch_size)):
        #     optimizer.zero_grad()
        #     for example_index in range(minibatch_size):
        #         input_vector, gold_label = validation_samples[minibatch_index * minibatch_size + example_index]
        N = len(validation_samples)
        optimizer.zero_grad()
        for index in tqdm(range(N)):
            input_vector, gold_label = validation_samples[index]
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            correct += int(predicted_label == gold_label)
            total += 1
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

    current = os.curdir
    models = os.path.join(current, 'experimental_models')
    PATH = os.path.join(models, name + '.pt')
    torch.save(model.state_dict(), PATH)
        # while not stopping_condition: # How will you decide to stop training and why
        # 	optimizer.zero_grad()
        # 	# You will need further code to operationalize training, ffnn.py may be helpful
        #
        # 	predicted_vector = model(input_vector)
        # 	predicted_label = torch.argmax(predicted_vector)
        # 	# You may find it beneficial to keep track of training accuracy or training loss;

        # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

        # You will need to validate your model. All results for Part 3 should be reported on the validation set.
        # Consider ffnn.py; making changes to validation if you find them necessary
