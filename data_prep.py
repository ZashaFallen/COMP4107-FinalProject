# Notes from trying to figure out data processing:
# Sources:
# https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets/cornell_corpus/data.py
# https://github.com/Currie32/Chatbot-from-Movie-Dialogue/blob/master/Chatbot_Attention.py
# The goal will be to get 4 files, encoder and decoder inputs for both training and testing.

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist

import numpy as np
import tensorflow as tf
import re

# Load the raw data
def get_raw_data():
    lines = open('raw_data/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conv_lines = open('raw_data/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

    # This should just print the lines and not actually reshape them, right?
    # The sentences that we will be using to train our model.
    lines[:10]
    # The sentences' ids, which will be processed to become our input and target data.
    conv_lines[:10]
    return lines, conv_lines


# Create a dictionary to map each line's id with its line
def map_IDs_to_lines(lines):
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line


# Create a list of all of the conversations' lines' ids.
def gather_conversation_IDs(conv_lines):
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    convs[:10]
    return convs


# Sort the sentences into questions (inputs) and answers (targets)
def separate_questions_answers(convs, id2line):
    questions = []
    answers = []
    for conv in convs:
        for i in range(len(conv)-1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i+1]])
    return questions, answers


def verifyData(questions, answers):
    limit = 0
    for i in range(limit, limit+5):
        print(questions[i])
        print(answers[i])
        print()
    print(len(questions))
    print(len(answers))
    print()


def filter_data(questions, answers):
    def filter_line(line):
        return ''.join([ ch for ch in line if ch in EN_WHITELIST ])
    def clean_line(line):
        line = line.lower()
        line = re.sub(r"i'm", "i am", line)
        line = re.sub(r"he's", "he is", line)
        line = re.sub(r"she's", "she is", line)
        line = re.sub(r"it's", "it is", line)
        line = re.sub(r"that's", "that is", line)
        line = re.sub(r"what's", "that is", line)
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

    # Remove punctuation
    questions = [ filter_line(line) for line in questions ]
    answers = [ filter_line(line) for line in answers ]
    # Remove contractions
    questions = [ clean_line(line) for line in questions ]
    answers = [ clean_line(line) for line in answers ]
    # Remove sentences that are too long/too short
    questions = [ line for line in questions if len(line.split) > 2 || len(line.split) < 15]
    answers = [ line for line in answers if len(line.split) > 2 || len(line.split) < 15]

    return questions, answers

def prep_data():
    lines, conv_lines = get_raw_data()
    id2line = map_IDs_to_lines(lines)
    convs = gather_converstation_IDs(conv_lines)
    questions, answers = separate_questions_answers(convs, id2line)
    verifyData(questions, answers)

    filtered_questions, filtered_answers = filter_data(questions, answers)

    # create a vocab dictionary which holds the frequency of words
    # remove rare words

    # create two dictionaries (questions and answers) mapping each vocab word to a unique integer
    # Add unique tokens to the dictionaries
    # Create inverse int -> vocab dictionaries

    # Add the end of sentence token to the end of every answer.
    # ...
