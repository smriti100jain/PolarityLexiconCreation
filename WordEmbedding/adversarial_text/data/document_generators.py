# Copyright 2017 Google, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Input readers and document/token generators for datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import csv
import os
import random

import tensorflow as tf
import data_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', '', 'Which dataset to generate data for')

# Preprocessing config
flags.DEFINE_boolean('output_unigrams', True, 'Whether to output unigrams.')
flags.DEFINE_boolean('output_bigrams', False, 'Whether to output bigrams.')
flags.DEFINE_boolean('output_char', False, 'Whether to output characters.')
flags.DEFINE_boolean('lowercase', True, 'Whether to lowercase document terms.')

# IMDB
flags.DEFINE_string('imdb_input_dir', '', 'The input directory containing the '
                    'IMDB sentiment dataset.')
flags.DEFINE_integer('imdb_validation_pos_start_id', 10621, 'File id of the '
                     'first file in the pos sentiment validation set.')
flags.DEFINE_integer('imdb_validation_neg_start_id', 10625, 'File id of the '
                     'first file in the neg sentiment validation set.')

# Yelp ABSA
flags.DEFINE_string('yelpabsa_input_dir', '', 'The input directory containing the '
                    'Yelp ABSA sentiment dataset.')
flags.DEFINE_integer('yelpabsa_validation_pos_start_id', 10621, 'File id of the '
                     'first file in the pos sentiment validation set.')
flags.DEFINE_integer('yelpabsa_validation_neg_start_id', 10625, 'File id of the '
                     'first file in the neg sentiment validation set.')


Document = namedtuple('Document',
                      'content is_validation is_test label add_tokens')


def documents(dataset='train',
              include_unlabeled=False,
              include_validation=False):
  """Generates Documents based on FLAGS.dataset.

  Args:
    dataset: str, identifies folder within IMDB data directory, test or train.
    include_unlabeled: bool, whether to include the unsup directory. Only valid
      when dataset=train.
    include_validation: bool, whether to include validation data.

  Yields:
    Document

  Raises:
    ValueError: if include_unlabeled is true but dataset is not 'train'
  """

  if include_unlabeled and dataset != 'train':
    raise ValueError('If include_unlabeled=True, must use train dataset')

  # Set the random seed so that we have the same validation set when running
  # gen_data and gen_vocab.
  random.seed(302)

  ds = FLAGS.dataset
  if ds == 'imdb':
    docs_gen = imdb_documents
  elif ds == 'yelpabsa':
    docs_gen = yelpabsa_documents
  elif ds == 'dbpedia':
    docs_gen = dbpedia_documents
  elif ds == 'rcv1':
    docs_gen = rcv1_documents
  elif ds == 'rt':
    docs_gen = rt_documents
  else:
    raise ValueError('Unrecognized dataset %s' % FLAGS.dataset)

  for doc in docs_gen(dataset, include_unlabeled, include_validation):
    yield doc


def tokens(doc):
  """Given a Document, produces character or word tokens.

  Tokens can be either characters, or word-level tokens (unigrams and/or
  bigrams).

  Args:
    doc: Document to produce tokens from.

  Yields:
    token

  Raises:
    ValueError: if all FLAGS.{output_unigrams, output_bigrams, output_char}
      are False.
  """
  if not (FLAGS.output_unigrams or FLAGS.output_bigrams or FLAGS.output_char):
    raise ValueError(
        'At least one of {FLAGS.output_unigrams, FLAGS.output_bigrams, '
        'FLAGS.output_char} must be true')

  content = doc.content.strip()
  if FLAGS.lowercase:
    content = content.lower()

  if FLAGS.output_char:
    for char in content:
      yield char

  else:
    tokens_ = data_utils.split_by_punct(content)
    for i, token in enumerate(tokens_):
      if FLAGS.output_unigrams:
        yield token

      if FLAGS.output_bigrams:
        previous_token = (tokens_[i - 1] if i > 0 else data_utils.EOS_TOKEN)
        bigram = '_'.join([previous_token, token])
        yield bigram
        if (i + 1) == len(tokens_):
          bigram = '_'.join([token, data_utils.EOS_TOKEN])
          yield bigram


def imdb_documents(dataset='train',
                   include_unlabeled=False,
                   include_validation=False):
  """Generates Documents for IMDB dataset.

  Data from http://ai.stanford.edu/~amaas/data/sentiment/

  Args:
    dataset: str, identifies folder within IMDB data directory, test or train.
    include_unlabeled: bool, whether to include the unsup directory. Only valid
      when dataset=train.
    include_validation: bool, whether to include validation data.

  Yields:
    Document

  Raises:
    ValueError: if FLAGS.imdb_input_dir is empty.
  """
  if not FLAGS.imdb_input_dir:
    raise ValueError('Must provide FLAGS.imdb_input_dir')

  tf.logging.info('Generating IMDB documents...')

  def check_is_validation(filename, class_label):
    if class_label is None:
      return False
    if('_' not in filename):
      return False
    file_idx = int(filename.split('_')[0])
    is_pos_valid = (class_label and
                    file_idx >= FLAGS.imdb_validation_pos_start_id)
    is_neg_valid = (not class_label and
                    file_idx >= FLAGS.imdb_validation_neg_start_id)
    return is_pos_valid or is_neg_valid

  dirs = [(dataset + '/pos', True), (dataset + '/neg', False)]
  if include_unlabeled:
    dirs.append(('train/unsup', None))

  for d, class_label in dirs:
    for filename in os.listdir(os.path.join(FLAGS.imdb_input_dir, d)):
      is_validation = check_is_validation(filename, class_label)
      if is_validation and not include_validation:
        continue

      with open(os.path.join(FLAGS.imdb_input_dir, d, filename)) as imdb_f:
        content = imdb_f.read()
      yield Document(
          content=content,
          is_validation=is_validation,
          is_test=False,
          label=class_label,
          add_tokens=True)



def yelpabsa_documents(dataset='train',
                   include_unlabeled=False,
                   include_validation=False):
  """Generates Documents for IMDB dataset.

  Data from http://ai.stanford.edu/~amaas/data/sentiment/

  Args:
    dataset: str, identifies folder within IMDB data directory, test or train.
    include_unlabeled: bool, whether to include the unsup directory. Only valid
      when dataset=train.
    include_validation: bool, whether to include validation data.

  Yields:
    Document

  Raises:
    ValueError: if FLAGS.yelpabsa_input_dir is empty.
  """
  if not FLAGS.yelpabsa_input_dir:
    raise ValueError('Must provide FLAGS.yelpabsa_input_dir')

  tf.logging.info('Generating YelpAbsa documents...')

  def check_is_validation(filename, class_label):
    if class_label is None:
      return False
    file_idx = int(filename.split('.')[0].split(' ')[0])
    is_pos_valid = (class_label and
                    file_idx >= FLAGS.yelpabsa_validation_pos_start_id)
    is_neg_valid = (not class_label and
                    file_idx >= FLAGS.yelpabsa_validation_neg_start_id)
    return is_pos_valid or is_neg_valid

  dirs = [(dataset + '/pos', True), (dataset + '/neg', False)]
  if include_unlabeled:
    dirs.append(('train/unsup', None))

  for d, class_label in dirs:
    for filename in os.listdir(os.path.join(FLAGS.yelpabsa_input_dir, d)):
      is_validation = check_is_validation(filename, class_label)
      if is_validation and not include_validation:
        continue

      with open(os.path.join(FLAGS.yelpabsa_input_dir, d, filename)) as yelpabsa_f:
        content = yelpabsa_f.read()
      yield Document(
          content=content,
          is_validation=is_validation,
          is_test=False,
          label=class_label,
          add_tokens=True)

 


