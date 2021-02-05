"""
Neural seq2seq in PyTorch
- some architecture from https://github.com/omarsar/pytorch_neural_machine_translation_attention
"""

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import time

from google.colab import drive
drive.mount('/content/drive')

lang = 'lat_test'
trainfeats = 'N;ABL;PL'

f = open(f'/content/drive/My Drive/Linguistics/Morpheme_Ordering/{lang}/{lang}.txt', encoding='UTF-8').read().strip().split('\n')
lines = f
original_word_pairs = [[w for w in l.split('\t')] for l in lines]
data = (pd.DataFrame(original_word_pairs, columns=["lemma", "form", "feats"])).dropna()

# for preprocessing (adds start and end tokens, etc.)
def preprocess_lemma(w):    
    w = w.lower()
    w_chars = ' '.join(list(w))
    w_chars = '<start> ' + w_chars
    return w_chars

def preprocess_feats(w):
    # no w.lower() since feats should be distinguished from lemma through case
    w_list = w.split(';')
    w_list.pop(0)
    w_chars = ' '.join(w_list)
    w_chars = ' ' + w_chars + ' <end>'
    return w_chars

def preprocess_form(w):    
    w = w.lower()
    w_chars = ' '.join(list(w))
    w_chars = '<start> ' + w_chars + ' <end>'
    return w_chars

# preprocesses data
data["lemma"] = data.lemma.apply(lambda w: preprocess_lemma(w)) + data.feats.apply(lambda w: preprocess_feats(w))
data["form"] = data.form.apply(lambda w: preprocess_form(w))

# dataframe with forms that match features
testdata = data[data.feats == trainfeats]
# for syncretism in training data
sync_forms = []
for index, row in testdata.iterrows():
    sync_forms.append(row[1])

# dataframe with forms that don't match features
traindata = data[data.feats != trainfeats]
# creates POS column, gets rid of rows that don't match POS
traindata['POS'] = traindata['feats'].astype(str).str[0]
traindata = traindata[traindata.POS == (trainfeats.split(';'))[0]]
# removes forms that are syncretic
traindata = traindata[~traindata['form'].isin(sync_forms)]

# creates dictionaries that map words to ids and vice versa
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        
        self.create_index()
        
    def create_index(self):
        for phrase in self.lang:
            # update with individual tokens
            self.vocab.update(phrase.split(' '))
            
        # sorts vocab
        self.vocab = sorted(self.vocab)

        # adds padding token (0)
        self.word2idx['<pad>'] = 0
        
        # word to id
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # +1 because of pad token
        
        # id to word
        for word, index in self.word2idx.items():
            self.idx2word[index] = word

# creates input and target languages from dataframe
inp_lang = LanguageIndex(traindata["lemma"].values.tolist())
targ_lang = LanguageIndex(traindata["form"].values.tolist())

# converts inputs and targets to tensors
input_tensor = [[inp_lang.word2idx[s] for s in lemma.split(' ')]  for lemma in traindata["lemma"].values.tolist()]
target_tensor = [[targ_lang.word2idx[s] for s in form.split(' ')]  for form in traindata["form"].values.tolist()]

# get max length of input and target tensors
def max_length(tensor):
    return max(len(t) for t in tensor)

max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

# padding
def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded

input_tensor = [pad_sequences(x, max_length_inp) for x in input_tensor]
target_tensor = [pad_sequences(x, max_length_tar) for x in target_tensor]

# training and val split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# helper class for data
class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x,y,x_len
    
    def __len__(self):
        return len(self.data)

# HYPERPARAMETERS: batch size, embedding dim, units, learning rate
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)
learning_rate = 0.001

# creates training and validation datasets
train_dataset = MyData(input_tensor_train, target_tensor_train)
val_dataset = MyData(input_tensor_val, target_tensor_val)

dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last=True, shuffle=True)

# encoder model
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.enc_units)
        
    def forward(self, x, lens, device):
        x = self.embedding(x) 
        x = pack_padded_sequence(x, lens)

        self.hidden = self.initialize_hidden_state(device)

        output, self.hidden = self.gru(x, self.hidden)
        
        output, _ = pad_packed_sequence(output)
        
        return output, self.hidden

    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_sz, self.enc_units)).to(device)

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, enc_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim + self.enc_units, 
                          self.dec_units,
                          batch_first=True)
        self.fc = nn.Linear(self.enc_units, self.vocab_size)
        
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.enc_units, 1)
    
    def forward(self, x, hidden, enc_output):
        enc_output = enc_output.permute(1,0,2)

        hidden_with_time_axis = hidden.permute(1, 0, 2)
        
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        
        attention_weights = torch.softmax(self.V(score), dim=1)
        
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        
        x = self.embedding(x)
        
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        
        output, state = self.gru(x)
        
        output =  output.view(-1, output.size(2))
        
        x = self.fc(output)
        
        return x, state, attention_weights
    
    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.dec_units))

criterion = nn.CrossEntropyLoss()

def loss_function(real, pred):
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    
    loss_ = criterion(pred, real) * mask
    return torch.mean(loss_)

# initializes device, encoder, decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, units, BATCH_SIZE)

encoder.to(device)
decoder.to(device)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

MAX_EPOCHS = 10
prev_loss = 1000000000.0 # arbitrarily large

for epoch in range(MAX_EPOCHS):
    start = time.time()
    
    encoder.train()
    decoder.train()
    
    total_loss = 0
    
    for (batch, (inp, targ, inp_len)) in enumerate(dataset):
        loss = 0
        
        xs, ys, lens = sort_batch(inp, targ, inp_len)
        enc_output, enc_hidden = encoder(xs.to(device), lens, device)
        dec_hidden = enc_hidden
        
        # teacher forcing
        dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * BATCH_SIZE)
        
        for t in range(1, ys.size(1)):
            predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                         dec_hidden.to(device), 
                                         enc_output.to(device))
            loss += loss_function(ys[:, t].to(device), predictions.to(device))
            dec_input = ys[:, t].unsqueeze(1)
            
        
        batch_loss = (loss / int(ys.size(1)))
        total_loss += batch_loss
        
        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()
        
        if batch % 100 == 0:
            print(f'Epoch {epoch + 1}, Batch {batch}, Loss {batch_loss.detach().item()}')
        
    avg_loss = total_loss/N_BATCH
    print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')
    print(f'Time taken for 1 epoch: {time.time()-start} sec\n')
    if (avg_loss > prev_loss):
        print(f'Ended on Epoch {epoch+1}.')
        break
    prev_loss = avg_loss

""" 
- Evaluate and get_fusion functions are still WIP
- Need to fix the encoder and decoder steps (and make sure everything has been
                converted properly to torch/numpy)

# gets the most likely output given features
def evaluate(features):
    attention_plot = np.zeros((max_length_tar, max_length_inp))
    # preprocessing (create list based on features split by space)
    inputs = [inp_lang.word2idx[i] for i in features.split(' ')]
    inputs = pad_sequences(inputs, max_length_inp)
    # turns sequence into tensor
    inputs = torch.from_numpy(inputs)

    result = ''

    hidden = [np.zeros((1, units))]

    # issue is here, need to
    # enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = torch.unsqueeze([targ_lang.word2idx['<start>']], 0)

    # iterate through maximum possible length of output
    for t in range(max_length_tar):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                        dec_hidden, enc_out)
        
        attention_weights = torch.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        # gets most likely id
        predicted_id = torch.argmax(predictions[0]).numpy()

        # return result if the token is the end token
        if targ_lang.idx2word[predicted_id] == '<end>':
            return result
        # convert id to token, then add to result
        else:
            result += targ_lang.idx2word[predicted_id]

        # feeds predicted id back into model
        dec_input = torch.unsqueeze([predicted_id], 0)

    return result, attention_plot

def get_fusion(features, word):
    attention_plot = np.zeros((max_length_tar, max_length_inp))

    # converts word to chars
    wordlist = list(word)
    wordlist.append('<end>')
    
    inputs = [inp_lang.word2idx[i] for i in features.split(' ')]
    inputs = pad_sequences(inputs, max_length_inp)
    
    # turns sequence into tensor
    inputs = torch.from_numpy(inputs)

    word_surprisal = 0

    hidden = [np.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs.to(device), hidden, device)

    dec_hidden = enc_hidden
    dec_input = torch.unsqueeze([targ_lang.word2idx['<start>']], 0)

    # iterate through word character by character
    for t in range(len(wordlist)-1):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                        dec_hidden, enc_out)
        
        attention_weights = torch.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        # gets character index
        word_id = targ_lang.word2idx[wordlist[t]]
        logits_arr = np.asarray(predictions[0])

        # turns logits into probs w/ softmax, gets prob of token
        probs_arr = (torch.softmax(logits_arr, dim=1)).numpy()
        prob_char = probs_arr[word_id]

        # adds to surprisal
        word_surprisal += -np.log2(prob_char)
        
        if wordlist[t] == '<end>':
            return word_surprisal, attention_plot
        
        # feeds predicted id back into model
        dec_input = torch.unsqueeze([word_id], 0)

    return word_surprisal, attention_plot

def plot_attention(attention, features, form):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='binary')
    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + features, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + form, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def get_att_plot(features, form):
    surprisal, attention_plot = get_fusion(features, form)
    form_list = list(form)
    form_list.append('<end>')
    form_chars = ' '.join(form_list)
    attention_plot = attention_plot[:len(form_chars.split(' ')), :len(features.split(' '))]
    plot_attention(attention_plot, features.split(' '), form_chars.split(' '))

testing_feats = '<start> g r a v e d o ABL PL <end>'
testing_form = 'gravēdinibus'
get_att_plot(testing_feats, testing_form)
surp, att = get_fusion(testing_feats, testing_form)
result = evaluate(testing_feats)
get_att_plot(testing_feats, result)
surp1, att2 = get_fusion(testing_feats, result)
print('Surprisal of correct form:', surp)
print('Expected Form:', result)
print('Surprisal of expected form:', surp1)
"""