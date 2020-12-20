# basic imports
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import re
import numpy as np
import os
import io
import time
import csv
import statistics

# specify features for model
trainfeats = ['N', 'DAT', 'PL']

# file path
filepath = '/content/drive/My Drive/Linguistics/Morpheme_Ordering'

# choose language
language = 'lat'

# paths to language and features
languagefile_io = os.path.join(filepath, language, f'{language}.txt')
featsfile_io = os.path.join(filepath, language, 'featuresfile.txt')

# creates file with all feature combinations in language
with open(languagefile_io, 'r') as languagefile:
    langreader = csv.reader(languagefile, delimiter='\t')
    featsfile = open(featsfile_io, 'w')
    rows_seen = set()

    for row in langreader:
        if not row: # empty rows
            continue
        
        # if the features are not in the set
        if row[2] not in rows_seen:
            # add features to file and to set
            featsfile.write(f'{row[2]}\n')
            rows_seen.add(row[2])

    featsfile.close()

# every time a file is opened it becomes an _io.TextIOWrapper
# so files need to be converted back to strings
languagefile = str(languagefile_io)
featsfile = str(featsfile_io)

# file path for model
modelfile = '_'.join(trainfeats)
modelpath = os.path.join(filepath, language, modelfile)
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

trainfile = os.path.join(filepath, language, modelfile, f'{modelfile}_train.txt')
testfile_io = os.path.join(filepath, language, modelfile, f'{modelfile}_test.txt')
resultsfile_io = os.path.join(filepath, language, modelfile, f'{modelfile}_results.txt')

# directory for models
checkpoint_dir = os.path.join(filepath, 'saved_models', language, f'{modelfile}_weights/')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# generates a training file from unimorph data
# only trains on subsets with numfeats or more combinations, not including lemma
def subsets(infile, trainfile, testfile, features, numfeats):
    with open(infile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        trainset = open(trainfile, 'w')
        testset = open(testfile, 'w')

        for row in csvreader:
            if not row:
                continue

            # splits annotations at ;
            rowfeats = row[2].split(';')

            if (rowfeats[0] != features[0]):
                continue

            # creates set versions
            rowset = set(rowfeats)
            featset = set(features)

            # gets rid of multi-word forms
            if ' ' in row[0]:
                continue

            # turns words/features into strings
            charword = ' '.join(list(row[1]))
            rowfeats_string = ' '.join(rowfeats)
            lemma_list = list(row[0])
            lemma_string = ' '.join(lemma_list)

            # if they are not equal, but do intersect, add to training set
            if ((rowset != featset) and (len(rowset & featset) >= numfeats)):
                trainset.write(f'<start> {charword} <end>\t<start> {rowfeats_string} {lemma_string} <end>\n')
            if (rowset == featset):
                testset.write(f'<start> {charword} <end>\t<start> {rowfeats_string} {lemma_string} <end>\n')
        trainset.close()
        testset.close()

# generates morphpair dataset (form, feats)
def morphpairs(infile, n):
    lines = io.open(infile, encoding='UTF-8').read().strip().split('\n')
    pairs = [[w for w in l.split('\t')] for l in lines[:n]]
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

languagefile = str(languagefile_io)
featsfile = str(featsfile_io)

# create training and test files
subsets(languagefile, trainfile, testfile_io, trainfeats, 2)

# creates morphpair dataset from training and test files
lang, feats = morphpairs(trainfile, None)
lang_test, feats_test = morphpairs(testfile_io, None)

# loads dataset, creates tensors
intensor, targtensor, infeats, targmorphs = load_dataset(trainfile, None)
intensor_test, targtensor_test, infeats_test, targmorphs_test = load_dataset(testfile_io, None)
max_length_targ, max_length_in = targtensor.shape[1], intensor.shape[1]

# create validation set w/ 80-20 split
intensor_train, intensor_val, targtensor_train, targtensor_val = train_test_split(intensor, targtensor, test_size=0.2)

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

# attention (Bahdanau et al., 2015)
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
checkpoint_prefix = os.path.join(checkpoint_dir, 'checkpoint')
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

# training function
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targmorphs.word_index['<start>']] * BATCH_SIZE, 1)

        # teacher forcing
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

EPOCHS = 10 # max epochs
losses_y = []
total_loss = 1000000.0 # arbitrary large number

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    prev_loss = total_loss
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        losses_y.append(batch_loss.numpy())
        total_loss += batch_loss

        if batch % 100 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch}, Loss: {batch_loss.numpy()}')
    
    # early stopping
    percent_loss = 1 - total_loss/prev_loss
    if (percent_loss <= 0.2):
        print(f'Stopped training on Epoch {epoch+1}, with {percent_loss*100}% difference in loss from Epoch {epoch}\nAverage Loss: {total_loss / steps_per_epoch}')
        break
    
    print(f'Epoch: {epoch+1}, Average Loss: {total_loss / steps_per_epoch}')
    print(f'Time taken for epoch: {time.time() - start} sec\n')

# saving the model (saves last weights only)
checkpoint.save(file_prefix = checkpoint_prefix)

# generate loss plot
plt.style.use('seaborn-whitegrid')
n = 100
plt.plot(losses_y[::n], 'o', color='black')

# gets the most likely output given features
def evaluate(features):
    # preprocessing (create list based on features split by space)
    inputs = [infeats.word_index[i] for i in features.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                        maxlen=max_length_in, padding='post')
    # turns sequence into tensor
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targmorphs.word_index['<start>']], 0)

    # iteratre through maximum possible length of output
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                        dec_hidden, enc_out)

        # gets most likely id
        predicted_id = tf.argmax(predictions[0]).numpy()

        # return result if the token is the end token
        if targmorphs.index_word[predicted_id] == '<end>':
            return result
        # convert id to token, then add to result
        else:
            result += targmorphs.index_word[predicted_id]

        # feeds predicted id back into model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result

def get_fusion(features, word):
    softmax = tf.keras.layers.Softmax()

    # converts word to chars
    wordlist = list(word)
    wordlist.append('<end>')
    
    inputs = [infeats.word_index[i] for i in features.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                        maxlen=max_length_in, padding='post')
    
    # turns sequence into tensor
    inputs = tf.convert_to_tensor(inputs)

    word_surprisal = 0

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targmorphs.word_index['<start>']], 0)

    # iterate through word character by character
    for t in range(len(wordlist)-1):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                        dec_hidden, enc_out)

        # gets character index
        word_id = targmorphs.word_index[wordlist[t]]
        logits_arr = np.asarray(predictions[0])

        # turns logits into probs w/ softmax, gets prob of token
        probs_arr = softmax(logits_arr).numpy()
        prob_char = probs_arr[word_id]

        # adds to surprisal
        word_surprisal += -np.log2(prob_char)

        if word[t] == '<end>':
            return word_surprisal

        # feeds predicted id back into model
        dec_input = tf.expand_dims([word_id], 0)

    return word_surprisal

# restores model
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

testfile = str(testfile_io)
resultsfile = str(resultsfile_io)

mean_surp_list = []
med_surp_list = []
sstdev_surp_list = []
pstdev_surp_list = []

with open(testfile, 'r') as testfile:
    testreader = csv.reader(testfile, delimiter='\t')
    resultsfile = open(resultsfile, 'w')

    resultsfile.write('Correct Form\tPredicted Form\tSurprisal\n')

    fusion_list = []

    i = 0
    start = time.time()

    for row in testreader:
        if not row:
            continue

        i += 1
        correct_list = row[0].split(' ')

        # remove start and end tokens
        correct_list.pop(0)
        correct_list.pop()

        correct_form = ''.join(correct_list)
        correct_form = correct_form.lower()
        
        features_list = row[1].split(' ')
        featstring = ' '.join(features_list)
        featstring = featstring.lower()

        expected_form = evaluate(featstring)
        form_surp = get_fusion(featstring, correct_form)
        
        resultsfile.write(f'{correct_form}\t{expected_form}\t{form_surp}\n')
        if (i % 1000 == 0):
            print(f'Row {i}, word {correct_form}, surprisal: {form_surp}, time taken: {time.time() - start}')
        
        fusion_list.append(form_surp)

    mean_surp_form = statistics.mean(fusion_list)
    med_surp_form = statistics.median(fusion_list)
    sstdev_surp_form = statistics.stdev(fusion_list)
    pstdev_surp_form = statistics.pstdev(fusion_list)

    mean_surp_list.append(mean_surp_form)
    med_surp_list.append(med_surp_form)
    sstdev_surp_list.append(sstdev_surp_form)
    pstdev_surp_list.append(pstdev_surp_form)

    resultsfile.close()

print(f'Mean: {mean_surp_form}, Median: {med_surp_form}, Sample: {sstdev_surp_form}, Pop: {pstdev_surp_form}')

# Language-wide surprisal
# Creates a file with mean and median for each combination in a language
# Run after training on all feature combinations (executed automatically in model_run.py)
surpfile = os.path.join(filepath, language, f'{language}_surprisals')
with open(featsfile, 'r') as featsfile:
    featsreader = csv.reader(featsfile, delimiter='\t')
    surpfile = open(surpfile, 'w')

    for i, row in enumerate(featsreader):
        if i > len(mean_surp_list):
            break
        surpfile.writerow(f'{row[0]}\t{mean_surp_form[i]}\t{med_surp_form[i]}\t{sstdev_surp_form[i]}\t{pstdev_surp_form[i]}\n')
    featsfile.close()