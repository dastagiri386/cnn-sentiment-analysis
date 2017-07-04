import numpy as np
import re
import itertools
import csv

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(data_file):
	f = open(data_file, 'r')
	reader = csv.reader(f)
	data = []
	flag = 1
	for row in reader:
		if flag == 1:
			flag = 0
			pass
		else:
			data.append(row)
	f.close()

	relevant_data = [[i[10], i[12], i[13], i[14], i[15], i[16], i[17], i[18], i[19]] for i in data]
	x_text = []
	y = []
	score_dict = {'app' : 4, 'sig' : 1, 'inv' : 2, 'inn' : 3, 'env' : 5}
	for item in relevant_data:
		if item[7] in score_dict and item[score_dict[item[7]]].isdigit(): # ignore:  overall criterion, cases when criteria scores are NA
			vec = [0, 0, 0, 0, 0, 0, 0, 0, 0]
			val = int(item[score_dict[item[7]]]) - 1
			vec[val] = 1
			text = item[8].split('.')
			for t in text:
				cleaned_txt = clean_str(t)
				if len(cleaned_txt) > 0:
					x_text.append(clean_str(t))
					y.append(vec)
	#print x_text, y_text, len(x_text), len(y)
	return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

