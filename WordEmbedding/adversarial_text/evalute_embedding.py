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

"""Evaluates text classification model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import time

import tensorflow as tf

import graphs
import pickle
import os
import gzip

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '',
                    'BNS name prefix of the Tensorflow eval master, '
                    'or "local".')
flags.DEFINE_string('eval_dir', '/tmp/text_eval',
                    'Directory where to write event logs.')
flags.DEFINE_string('eval_data', 'test', 'Specify which dataset is used. '
                    '("train", "valid", "test") ')

flags.DEFINE_string('checkpoint_dir', '/tmp/text_train',
                    'Directory where to read model checkpoints.')
flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run the eval.')
flags.DEFINE_integer('num_examples', 32, 'Number of examples to run.')
flags.DEFINE_bool('run_once', False, 'Whether to run eval only once.')
flags.DEFINE_string('saveEmbeddingPath','/tmp/text_eval','Save Path for saving embedding file.')

def restore_from_checkpoint(sess, saver):
  """Restore model from checkpoint.

  Args:
    sess: Session.
    saver: Saver for restoring the checkpoint.

  Returns:
    bool: Whether the checkpoint was found and restored
  """
  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  if not ckpt or not ckpt.model_checkpoint_path:
    tf.logging.info('No checkpoint found at %s', FLAGS.checkpoint_dir)
    return False

  saver.restore(sess, ckpt.model_checkpoint_path)
  return True


def run_eval(eval_ops, summary_writer, saver,embdings):
  """Runs evaluation over FLAGS.num_examples examples.

  Args:
    eval_ops: dict<metric name, tuple(value, update_op)>
    summary_writer: Summary writer.
    saver: Saver.

  Returns:
    dict<metric name, value>, with value being the average over all examples.
  """
  sv = tf.train.Supervisor(logdir=FLAGS.eval_dir, saver=None, summary_op=None)
  with sv.managed_session(
      master=FLAGS.master, start_standard_services=False) as sess:
    if not restore_from_checkpoint(sess, saver):
      return
    sv.start_queue_runners(sess)

    metric_names, ops = zip(*eval_ops.items())
    value_ops, update_ops = zip(*ops)

    # Run update ops
    num_batches = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    tf.logging.info('Running %d batches for evaluation.', num_batches)
    embeddings = sess.run(embdings)
    print(np.shape(embeddings))
    print(embeddings[0])
    f = gzip.open(os.path.join(FLAGS.saveEmbeddingPath,'embedding.pklz'),'wb')
    pickle.dump(embeddings[0],f)
    f.close()
    print('--------------------------------------------------------------------------------------')
    '''
    in order to open gzip file:
    f = gzip.open('testPickleFile.pklz','rb')
    myNewObject = pickle.load(f)
    f.close()
    '''    


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  tf.logging.info('Building eval graph...')
  output = graphs.get_model().eval_graph(FLAGS.eval_data)
  eval_ops, moving_averaged_variables, embdings = output

  saver = tf.train.Saver(moving_averaged_variables)
  summary_writer = tf.summary.FileWriter(
      FLAGS.eval_dir, graph=tf.get_default_graph())

  while True:
    run_eval(eval_ops, summary_writer, saver,embdings)
    if FLAGS.run_once:
      break
    time.sleep(FLAGS.eval_interval_secs)


if __name__ == '__main__':
  tf.app.run()
