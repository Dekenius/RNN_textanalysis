
# SECTIONS
# - Loading the Data
# - Preparing the Data
# - Building the Model
# - Training the Model
# - Fixing Custom Sentences
# - Summary



import pandas as pd
import numpy as np
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import time
import re
from sklearn.model_selection import train_test_split

import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Loading the Data


def load_book(path):
    """Load a book from its file"""
    input_file = os.path.join(path)
    with open(input_file, encoding='windows-1252') as f:
        book = f.read()
    return book


# Collect all of the book file names
path = './books/'
book_files = [f for f in listdir(path) if isfile(join(path, f))]
book_files = book_files[1:]



# Load the books using the file names
books = []
for book in book_files:
    books.append(load_book(path+book))


# Check to ensure the text looks alright
# print(books[0][:1000])
#
# print('\n\n')


# ## Preparing the Data

# In[10]:

def clean_text(text):
    '''Remove unwanted characters and extra spaces from the text'''
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','! ', text)
    text = re.sub('\?','? ', text)
    text = re.sub(' +',' ', text)
    return text


# In[11]:

# Clean the text of the books
clean_books = []
for book in books:
    clean_books.append(clean_text(book))


# Check to ensure the text looks alright
# print(clean_books[0][:1000])



# Create a dictionary to convert the vocabulary (characters) to integers
vocab_to_int = {}
count = 0
for book in clean_books:
    doc = nlp(book)
    for character in doc:
        character = character.text # spacy object -> string
        if character not in vocab_to_int:
            vocab_to_int[character] = count
            count += 1

# Add special tokens to vocab_to_int
codes = ['<PAD>','<EOS>','<GO>']
for code in codes:
    vocab_to_int[code] = count
    count += 1



# # Check the size of vocabulary and all of the values
# vocab_size = len(vocab_to_int)
# print("The vocabulary contains {} characters.".format(vocab_size))
# print(sorted(vocab_to_int))


# Create another dictionary to convert integers to their respective characters
int_to_vocab = {value : key for (key, value) in vocab_to_int.items()}
