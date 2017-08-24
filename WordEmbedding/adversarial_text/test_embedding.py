from __future__ import division
from __future__ import print_function
import numpy as np
import math
import time
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
import graphs
import pickle
import os
import gzip

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('saveEmbeddingPath','/tmp/text_eval','Save Path for saving embedding file.')
flags.DEFINE_string('vocabPath','/tmp/text_eval','Path for vocabulary of dataset.')


f = gzip.open(os.path.join(FLAGS.saveEmbeddingPath,'embedding.pklz'),'rb')
embedding = pickle.load(f)
f.close()
f = open(os.path.join(FLAGS.vocabPath,'vocab.txt'),'rb')
vocab = f.read().split('\n')
f.close()

posTerms = ["happy", "satisfied", "good", "positive", "excellent", "flawless"]
negTerms = ["unhappy", "unsatisfied", "bad", "negative", "poor"]
 
posTerms_id = []
for item in posTerms:
	if(item in vocab):
		posTerms_id.append(vocab.index(item))

negTerms_id = []
for item in negTerms:
	if(item in vocab):
		negTerms_id.append(vocab.index(item))
'''
posTerms_id = [vocab.index(item) for item in posTerms]
negTerms_id = [vocab.index(item) for item in negTerms]
'''


def getMostSimilar(term,threshold,topN):
    similar = []
    sim = cosine_similarity(embedding[term],embedding)
    indices = sim[0].argsort()[-topN:][::-1]
    a = np.where(sim[0][indices]>=threshold)
    return(indices[a])

def cosinesimilarity(w,term):
    sim = cosine_similarity(embedding[w],embedding[term])[0][0]
    return(sim)

target = open(os.path.join(FLAGS.saveEmbeddingPath,'mostSimilar.txt'),'w')
for i in posTerms_id+negTerms_id:
	target.write('Most Similar Words to '+ vocab[i] + '\n')
	Top10 = getMostSimilar(i,0,30)
	for j in Top10:
		target.write(vocab[j]+',')
	target.write('\n')
	target.write('-------------------------------------------------------')
	target.write('\n')

target.close()
