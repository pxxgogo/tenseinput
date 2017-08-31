# -*- coding: utf-8 -*-


import time
import os
import json

import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime

from provider import Provider

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class Model(object):
    """The RNN model."""

    def __init__(self, is_training, config):
        self.batch_size = config['batch_size']
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.input_channel = config['input_channel']
        self.model_structure = config['model_structure']

        self._input_data = tf.placeholder(data_type(), [self.batch_size, self.input_channel, self.input_size])
        self._targets = tf.placeholder(tf.int16, [self.batch_size, self.output_size])
        data = tf.reshape(self._input_data, [self.batch_size, self.input_channel, self.input_size, 1])
        for i, layer in enumerate(self.model_structure):
            data = self.add_conv_layer(
                No=i,
                input=data,
                filter_size=layer["filter_size"],
                out_channels=layer["out_channels"],
                filter_type=layer["filter_type"]
            )
            data = self.add_pool_layer(i, data, layer["pool_size"], [1, 1, 1, 1], pool_type=layer["pool_type"])

        #
        # print(data.shape)
        # exit()

        data = tf.reshape(data, [self.batch_size, -1])

        softmax_w = tf.get_variable(
            "softmax_w", [data.shape[1], self.output_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [self.output_size], dtype=data_type())
        self.logits = logits = tf.matmul(data, softmax_w) + softmax_b
        # print(self.logits.shape)
        self._cost = cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._targets, logits=logits))
        self._predict_op = tf.argmax(logits, 1)

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config['max_grad_norm'])
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def add_conv_layer(self, No, input, filter_size, out_channels, filter_type):
        with tf.variable_scope("conv_layer_%d" % No):
            W = tf.get_variable('filter', [filter_size[0], filter_size[1], input.shape[3], out_channels])
            b = tf.get_variable('bias', [out_channels])
            conv = tf.nn.conv2d(
                input,
                W,
                strides=[1, 1, 1, 1],
                padding=filter_type,
                name='conv'
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        return h

    def add_pool_layer(self, No, input, pool_size, strides, pool_type):
        for i in range(2):
            if pool_size[i] == -1:
                pool_size[i] = input.shape[1 + i]
        with tf.variable_scope("pool_layer_%d" % No):
            pooled = tf.nn.max_pool(
                input,
                ksize=[1, pool_size[0], pool_size[1], 1],
                padding=pool_type,
                strides=strides,
                name='pool'
            )
        return pooled

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    # @property
    # def initial_state(self):
    #     return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict_op(self):
        return self._predict_op


def check_ans(tags, predict_vals):
    No = 0
    length = tags.shape[0]
    right_num = 0
    debug_val = 0
    # print(tags)
    # print(predict_vals)
    while No < length:
        tag = tags[No]
        predict_val = predict_vals[No]
        if tag[predict_val] == 1:
            right_num += 1
        else:
            if tag[0] == 1:
                debug_val += 1
        No += 1
    return right_num, debug_val


# @make_spin(Spin1, "Running epoch...")
def run_epoch(session, model, provider, status, eval_op, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    stage_time = time.time()
    costs = 0.0
    iters = 0
    right_num = 0
    debug_val = 0
    sum = 0
    provider.status = status
    for step, (x, y) in enumerate(provider()):
        fetches = [model.cost, model.predict_op, eval_op]
        if status == 'test':
            fetches.append(model.logits)
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        if status == 'test':
            cost, predict_val, _, logits = session.run(fetches, feed_dict)
            # print(y, logits)
        else:
            cost, predict_val, _ = session.run(fetches, feed_dict)
        right_num_sub, debug_val_sub = check_ans(y, predict_val)
        right_num += right_num_sub
        debug_val += debug_val_sub
        sum += y.shape[0]
        costs += cost
        iters += 1
        epoch_size = provider.get_epoch_size()
        divider_10 = epoch_size // 10
        if verbose and step % divider_10 == 0:
            print("%.3f speed: %.0f wps time cost: %.3fs precision: %.3f debug_value: %.3f" %
                  (step * 1.0 / epoch_size,
                   sum / (time.time() - start_time), time.time() - stage_time, right_num / sum, debug_val / sum))
            stage_time = time.time()

    return np.exp(costs / iters), right_num / sum, debug_val / sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', type=int, default=0)
    parser.add_argument('--model_dir', type=str, default="./model")
    args = parser.parse_args()
    model_dir = args.model_dir
    train_type = args.train_type
    provider = Provider("./config.json")
    provider.status = 'train'
    config = provider.get_config()
    eval_config = config.copy()
    eval_config['batch_size'] = 1
    if not os.path.exists("./model"):
        os.mkdir("./model")

    # print (config)
    # print (eval_config)
    session_config = tf.ConfigProto(log_device_placement=False)
    session_config.gpu_options.allow_growth = False
    with tf.Graph().as_default(), tf.Session(config=session_config) as session:
        initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = Model(is_training=False, config=eval_config)

        session.run(tf.global_variables_initializer())
        if train_type == 1:
            new_saver = tf.train.Saver()
            new_saver.restore(session, tf.train.latest_checkpoint(
                model_dir))
        for v in tf.global_variables():
            print(v.name)
        saver = tf.train.Saver()
        best_cost = 10000
        model_No = 0
        best_precision = 0
        with open("./model/model_config.json", 'w') as file_handle:
            file_handle.write(json.dumps(config))
        for i in range(config['max_max_epoch']):
            # lr_decay = config['lr_decay'] ** max(i - config['max_epoch'], 0.0)
            # m.assign_lr(session, config['learning_rate'] * lr_decay)
            m.assign_lr(session, config['learning_rate'])
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            print("Starting Time:", datetime.now())
            train_perplexity, precision, debug_val = run_epoch(session, m, provider, 'train', m.train_op, verbose=True)
            print("Ending Time:", datetime.now())
            print("Starting Time:", datetime.now())
            test_perplexity, precision, debug_val = run_epoch(session, mtest, provider, 'test', tf.no_op())
            print("COST:", test_perplexity)
            print("Test Precision: %.3f (%.3f)" % (precision, debug_val))
            if test_perplexity < best_cost:
                best_precision = precision
                best_cost = test_perplexity
                save_path = saver.save(session, './model/model', global_step=model_No)
                model_No += 1
                print("SAVE!!!!")
                print("Model saved in file: %s" % save_path)

            print("Ending Time:", datetime.now())
        print("BEST PRECISION", best_precision)


if __name__ == "__main__":
    main()
