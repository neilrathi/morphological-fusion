trainfeats = ['ADJ', 'VOC', 'NEUT', 'SG', 'gravedinosus'] # gravēdinōsum

trainingfile = "/lat/lattrainset.txt"
testingfile = "/lat/lattestset.txt"
languagefile = "/lat/lat.txt"

checkpoint_dir = '/lat/latmodel/' # sets checkpoint


# basic imports
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import csv

# generates a training file from unimorph data
# only trains on subsets with numfeats or more combinations, not including lemma
def subsets(infile, trainfile, testfile, features, numfeats):
    with open(infile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        trainset = open(trainfile, 'w')
        testset = open(testfile, 'w')

        # creates a special version of features w/o the lemma
        features_a = list(features)
        features_a.pop()

        for row in csvreader:
            if not row:
                continue

            # splits annotations at ;
            rowfeats = row[2].split(';')

            # creates set versions
            rowset = set(rowfeats)
            featset = set(features_a)

            # turns words/features into strings
            charword = ' '.join(list(row[1]))
            rowfeats_string = ' '.join(rowfeats)

            # if they are not equal, but do intersect, add to training set
            if ((rowset != featset) and (len(rowset & featset) >= numfeats)):
                trainset.write(f'<start> {charword} <end>,<start> {rowfeats_string} {row[0]} <end>\n')
            if (rowset == featset):
                testset.write(f'<start> {charword} <end>,<start> {rowfeats_string} {row[0]} <end>\n')
        trainset.close()
        testset.close()

# generates morphpair dataset (form, feats)
def morphpairs(infile, n):
    lines = io.open(infile, encoding='UTF-8').read().strip().split('\n')
    pairs = [[w for w in l.split(',')] for l in lines[:n]]
    return zip(*pairs)

# tokenizing
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer

def load_dataset(infile, n=None):
  # creating cleaned (input, output) pairs
  targmorphs, infeats = morphpairs(infile, n)

  intensor, intokenizer = tokenize(infeats)
  targtensor, targtokenizer = tokenize(targmorphs)

  return intensor, targtensor, intokenizer, targtokenizer

# create training and test files
subsets(languagefile, trainingfile, testingfile, trainfeats, 2)

# creates morphpair dataset from training and test files
lang, feats = morphpairs(trainingfile, None)
lang_test, feats_test = morphpairs(testingfile, None)

# loads dataset, creates tensors
intensor, targtensor, infeats, targmorphs = load_dataset(trainingfile, None)
intensor_test, targtensor_test, infeats_test, targmorphs_test = load_dataset(testingfile, None)
max_length_targ, max_length_in = targtensor.shape[1], intensor.shape[1]

# create validation set w/ 80-20 split
intensor_train, intensorval_val, targtensor_train, targtensor_val = train_test_split(intensor, targtensor, test_size=0.2)

BUFFER_SIZE = len(intensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(intensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(infeats.word_index)+1
vocab_tar_size = len(targmorphs.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((intensor_train, targtensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# encoder
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True,
                    return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

# attention
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# decoder
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights

# optimizer and loss
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# saves in checkpoints
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

# training function
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targmorphs.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy()}')
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print(f'Epoch {epoch+1} Loss {total_loss / steps_per_epoch}')
    print(f'Time taken for 1 epoch {time.time() - start} sec\n')

def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_in))
    
    inputs = [infeats.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                        maxlen=max_length_in, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targmorphs.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                        dec_hidden, enc_out)

        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targmorphs.index_word[predicted_id] + ' '

        if targmorphs.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input: %s' % (sentence))
  print(f'Predicted translation: {result}')

  attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]

# testing

featstring = f"<start> {(' '.join(trainfeats)).lower()} <end>"
translate(featstring)
