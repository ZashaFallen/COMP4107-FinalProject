# Sources:
# https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets/cornell_corpus/data.py
# https://github.com/Currie32/Chatbot-from-Movie-Dialogue/blob/master/Chatbot_Attention.py

EN_WHITELIST = '.0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
MAX_LINE_LENGTH = 16
MIN_LINE_LENGTH = 2

limit = {
        'maxq' : 25,
        'minq' : 2,
        'maxa' : 25,
        'mina' : 2
}

import itertools
import numpy as np
import nltk
import re
import pickle
from random import sample

'''
Load the raw data
'''
def get_raw_data():
    lines = open('raw_data/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conv_lines = open('raw_data/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

    # This should just print the lines and not actually reshape them, right?
    # The sentences that we will be using to train our model.
    lines[:10]
    # The sentences' ids, which will be processed to become our input and target data.
    conv_lines[:10]
    return lines, conv_lines


'''
Create a dictionary to map each line's id with its line
'''
def map_IDs_to_lines(lines):
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line


'''
Create a list of all of the conversations' lines' ids.
'''
def gather_conversation_IDs(conv_lines):
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    convs[:10]
    return convs


'''
Sort the sentences into questions (inputs) and answers (targets)
'''
def separate_questions_answers(convs, id2line):
    questions = []
    answers = []
    for conv in convs:
        for i in range(len(conv)-1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i+1]])
    return questions, answers


'''
Print a portion of the data, to try and verify it is correct
'''
def verifyData(questions, answers):
    limit = 0
    for i in range(limit, limit+5):
        print(questions[i])
        print(answers[i])
        print()
    print(len(questions))
    print(len(answers))
    print()


'''
filter and clean the dataset
    - Remove punctuation
    - Remove contractions
    - Remove sentences that are too long/too short
'''
def filter_data(questions, answers):
    def clean_line(line):
        line = line.lower()
        line = re.sub(r"i'm", "i am", line)
        line = re.sub(r"he's", "he is", line)
        line = re.sub(r"she's", "she is", line)
        line = re.sub(r"it's", "it is", line)
        line = re.sub(r"that's", "that is", line)
        line = re.sub(r"what's", "what is", line)
        line = re.sub(r"where's", "where is", line)
        line = re.sub(r"how's", "how is", line)
        line = re.sub(r"\'ll", " will", line)
        line = re.sub(r"\'ve", " have", line)
        line = re.sub(r"\'re", " are", line)
        line = re.sub(r"\'d", " would", line)
        line = re.sub(r"\'re", " are", line)
        line = re.sub(r"won't", "will not", line)
        line = re.sub(r"can't", "cannot", line)
        line = re.sub(r"n't", " not", line)
        line = re.sub(r"n'", "ng", line)
        line = re.sub(r"'bout", "about", line)
        line = re.sub(r"'til", "until", line)
        return line
    def filter_line(line):
        return ''.join([ ch for ch in line if ch in EN_WHITELIST ])

    # Remove contractions
    questions = [ clean_line(line) for line in questions ]
    answers = [ clean_line(line) for line in answers ]
    # Remove punctuation
    questions = [ filter_line(line) for line in questions ]
    answers = [ filter_line(line) for line in answers ]
    # Remove sentences that are too long/too short
    tmp_questions = []
    tmp_answers = []
    for i, q in enumerate(questions):
        a = answers[i]
        if ((len(q.split()) >= MIN_LINE_LENGTH and len(q.split()) <= MAX_LINE_LENGTH) and
                    (len(a.split()) >= MIN_LINE_LENGTH and len(a.split()) <= MAX_LINE_LENGTH)):
            tmp_questions.append(q)
            tmp_answers.append(a)

    return tmp_questions, tmp_answers


'''
Sort the data by word length. This might speed up training
'''
def sort_data(questions, answers):
    sorted_questions = []
    sorted_answers = []

    for length in range(1, MAX_LINE_LENGTH+1):
        for i in enumerate(questions):
            if len(i[1]) == length:
                sorted_questions.append(questions[i[0]])
                sorted_answers.append(answers[i[0]])

    return sorted_questions, sorted_answers


'''
Create a vocabulary frequency dictionary
    - prune the least common words
    - create index-to-word list
    - create word-to-index dictionary
    return index2word, word2index, freq_dist
(maps each word to a unique integer)
'''
def get_frequency_distribution(tokenized_lines):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_lines))
    # get vocabulary of 8000 most used words
    vocab = freq_dist.most_common(8000)
    # index2word
    index2word = ['_'] + ['unk'] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


'''
Filter the tokenized sentence lists.
    - filter based on number of unknowns (words not in vocabulary)
    - filter out the worst sentences
    return filtered sentences
'''
def filter_unk(q_tokenized, a_tokenized, w2idx):
    filtered_q, filtered_a = [], []

    for q_line, a_line in zip(q_tokenized, a_tokenized):
        unk_count_q = len([ word for word in q_line if word not in w2idx ])
        unk_count_a = len([ word for word in a_line if word not in w2idx ])
        if unk_count_a <= 2:
            if unk_count_q > 0:
                if unk_count_q/len(q_line) > 0.2:
                    pass
            filtered_q.append(q_line)
            filtered_a.append(a_line)

    return filtered_q, filtered_a


'''
create the final dataset :
    - convert list of items to arrays of indices
    - add zero padding
    return ( [array_en([indices]), array_ta([indices]) )
'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
replace words with indices in a sequence
    - replace with unknown if word not in lookup
    return [list of indices]
'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup['unk'])
    return indices + [0]*(maxlen - len(seq))


def prep_data():
    # read in the daraw data from the files
    lines, conv_lines = get_raw_data()
    # get a dictionary to map each line's id with its line
    id2line = map_IDs_to_lines(lines)
    # get a list of conversation IDs
    convs = gather_conversation_IDs(conv_lines)
    # get a questions (inputs) array and an answers (targets) array
    questions, answers = separate_questions_answers(convs, id2line)
    verifyData(questions, answers)
    # filter and clean the data
    questions, answers = filter_data(questions, answers)
    verifyData(questions, answers)
    # sort data by word length
    '''
    #questions, answers = sort_data(questions, answers)
    verifyData(questions, answers)

    # tokenize the data and get a word frequency dictionary
    q_tokenized = [ [word.strip() for word in line.split() if word] for line in questions ]
    a_tokenized = [ [word.strip() for word in line.split() if word] for line in answers ]
    idx2w, w2idx, freq_dist = get_frequency_distribution(q_tokenized + a_tokenized)

    # filter out sentences with too many unknowns
    q_tokenized, a_tokenized = filter_unk(q_tokenized, a_tokenized, w2idx)

    # zero pad and create the final dataset
    idx_q, idx_a = zero_pad(q_tokenized, a_tokenized, w2idx)
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    # save the dictionary metadata
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
    }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    '''

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')

    return metadata, idx_q, idx_a


'''
split data into training (70%), testing (15%) and validation (15%)
    return (trainX, trainY), (testX,testY), (validX,validY)

Source:
    https://github.com/suriyadeepan/practical_seq2seq/blob/master/data_utils.py::split_dataset()
'''
def split_dataset(x, y):
    ratio = [0.7, 0.15, 0.15]

    # number of examples
    data_len = len(x)
    lengths = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lengths[0]], y[:lengths[0]]
    testX, testY = x[lengths[0]:lengths[0]+lengths[1]], y[lengths[0]:lengths[0]+lengths[1]]
    validX, validY = x[-lengths[-1]:], y[-lengths[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)


'''
 generate batches, by random sampling a bunch of items
    yield (x_gen, y_gen)
'''
def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T
