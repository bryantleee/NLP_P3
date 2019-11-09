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

#Our stuff we imported
from gensim.models import Word2Vec
from collections import Counter
import numpy as np


unk = '<UNK>'


class RNN(nn.Module):
	def __init__(self, hidden_dimension_size, hidden_layers): # Add relevant parameters
		super(RNN, self).__init__()
		# Fill in relevant parameters
		# Ensure parameters are initialized to small values, see PyTorch documentation for guidance
		self.rnn = nn.RNN(128, hidden_dimension_size, hidden_layers, batch_first=True)
		self.Linear = nn.Linear(hidden_dimension_size, 5)
		self.softmax = nn.LogSoftmax()
		self.loss = nn.NLLLoss()
		

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)	

	def forward(self, input_, h): 
		#begin code
		output, h_n = self.rnn(input_, h)
		distribution = self.Linear(output[0][-1])
		predicted_vector = self.softmax(distribution) # Remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
		#end code
		return predicted_vector

# You may find the functions make_vocab() and make_indices from ffnn.py useful; you are free to copy them directly (or call those functions from this file)


def main(): # Add relevant parameters
	train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
	
	counts = Counter()
	review_list = []
	for t in train_data:
		review_list.append(t[0])
		counts.update(t[0])

	model = Word2Vec(review_list, size=128, min_count=1)

	#create list of word embeddings for each review
	all_embeddings = []
	for t in train_data:
		all_embeddings.append([model[word] for word in t[0]])
	reshaped_embeddings = [np.stack(review_embeddings, axis=0) for review_embeddings in all_embeddings]

	#reshape embeddings
	reshaped_embeddings = []
	for review_embeddings in all_embeddings:
		temp_array = np.stack(review_embeddings, axis=0)
		temp_array = np.expand_dims(temp_array, axis=0)
		reshaped_embeddings.append(torch.from_numpy(temp_array).shape)
    	

	

	# Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
	# Further, think about where the vectors will come from. There are 3 reasonable choices:
	# 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
	# 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
	# 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further. 
	# Option 3 will be the most time consuming, so we do not recommend starting with this

	# model = RNN() # Fill in parameters
	optimizer = optim.SGD(model.parameters()) 

	while not stopping_condition: # How will you decide to stop training and why
		optimizer.zero_grad()
		# You will need further code to operationalize training, ffnn.py may be helpful

		predicted_vector = model(input_vector)
		predicted_label = torch.argmax(predicted_vector)
		# You may find it beneficial to keep track of training accuracy or training loss; 

		# Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

		# You will need to validate your model. All results for Part 3 should be reported on the validation set. 
		# Consider ffnn.py; making changes to validation if you find them necessary

