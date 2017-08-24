from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import csv
import os
import random

import tensorflow as tf
import data_utils
import codecs
import gzip
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import unicodedata
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', '', 'Which dataset to generate data for')

# Preprocessing config
flags.DEFINE_boolean('output_unigrams', True, 'Whether to output unigrams.')
flags.DEFINE_boolean('output_bigrams', False, 'Whether to output bigrams.')
flags.DEFINE_boolean('output_char', False, 'Whether to output characters.')
flags.DEFINE_boolean('word2vec', True, 'Whether to lowercase document terms.')

flags.DEFINE_string('word2vecembeddingpath', '../../../../../yelpabsa','Directory for logs and checkpoints.')
#initialize vocabulary present in word2vec with weights of word2vec and rest by random uniform distribution between (-1,1)

reload(sys)
sys.setdefaultencoding('UTF8')
filename = 'adversarial_text/data/wordvectors_w2v800k.tsv.gz'

posTerms = ["happy", "satisfied", "good", "positive", "excellent", "flawless", "unimpaired"]
negTerms = ["unhappy", "unsatisfied", "bad", "negative", "poor", "flawed", "defective"]

vocab = []
embd = []
count = 0
nonnormal = 0
file_h = codecs.getreader("utf-8")(gzip.open(filename))
existingVocabfile = open(os.path.join(FLAGS.word2vecembeddingpath,'vocab.txt'),'rb')
existingVocab = ((existingVocabfile.read()).strip()).split('\n')

print(len(existingVocab))

for line in file_h:
    line = line.strip()
    word,vec = line.split('\t')
    if(len(word)>=2 and all(unicodedata.category(i)=='Ll' for i in word) ):
        vec = vec.split(' ')
        vocab.append(word)
        vec = map(float,vec[0:])
        embd.append(vec)
        count = count + 1
        print(count)
    else:
        nonnormal = nonnormal + 1

embedding = np.asarray(embd)

EmbeddingMatrix = np.zeros((len(existingVocab),300))

indices_from = []
indices_to = []
for i in range(len(existingVocab)):
    print(i)
    item = existingVocab[i]
    if(item in vocab):
        indices_from.append(vocab.index(item))
        indices_to.append(i)
    else:
		EmbeddingMatrix[i] = np.random.uniform(-1,1,300)

EmbeddingMatrix[indices_to] = embedding[indices_from] 
print(np.shape(EmbeddingMatrix))

np.savez(os.path.join(FLAGS.word2vecembeddingpath,'embedding.npz'),EmbeddingMatrix)