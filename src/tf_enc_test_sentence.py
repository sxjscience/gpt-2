#!/usr/bin/env python3

import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder


def encode_sentence(sentence, model_name, models_dir='models'):
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    context_tokens = [enc.encode(sentence)]
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        lm_output = model.model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)
        out = sess.run(lm_output, feed_dict={context: context_tokens})
        return out['logits']

if __name__ == '__main__':
    sentence = ' natural language processing tools such as gluonnlp and torchtext'
    logits = encode_sentence(sentence, '117M')
    np.save('117M_gt_logits.npy', logits)
    logits = encode_sentence(sentence, '345M')
    np.save('345M_gt_logits.npy', logits)
