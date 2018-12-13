# Notes from trying to figure out data processing:
# Sources:
# https://github.com/suriyadeepan/practical_seq2seq/blob/master/datasets/cornell_corpus/data.py
# https://github.com/Currie32/Chatbot-from-Movie-Dialogue/blob/master/Chatbot_Attention.py
# The goal will be to get 4 files, encoder and decoder inputs for both training and testing.

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist

import numpy as np
import tensorflow as tf

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


# Create a dictionary to map each line's id with its text
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

'''
remove anything that isn't in the vocabulary
return str(pure en)
'''
def filter_data(questions, answers):
    def filter_line(line):
        return ''.join([ ch for ch in line if ch in EN_WHITELIST ])

    questions = [ filter_line(line) for line in questions ]
    answers = [ filter_line(line) for line in answers ]

    return questions, answers

def prep_data():
    lines, conv_lines = get_raw_data()
    id2line = map_IDs_to_lines(lines)
    convs = gather_converstation_IDs(conv_lines)
    questions, answers = separate_questions_answers(convs, id2line)
    verifyData(questions, answers)

    clean_questions, clean_answers = filter_data(questions, answers)
    # filter/clean data
        # remove unwanted characters
        # alter word format
        # filter out too long/short sentences
    # create a vocab dictionary which holds the frequency of words
    # remove rare words

    # create two dictionaries (questions and answers) mapping each vocab word to a unique integer
    # Add unique tokens to the dictionaries
    # Create inverse int -> vocab dictionaries

    # Add the end of sentence token to the end of every answer.
    # ...
