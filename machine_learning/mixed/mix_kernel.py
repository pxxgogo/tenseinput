# -*- coding: utf-8 -*-


import time
import os
import json

import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as pl

from mix_provider import Provider
from mix_model import Model

fig, axes = pl.subplots(2, 1, sharex=True)
# print(axes)
# exit()
# ax = [axes[0].add_subplot(1, 1, 1), axes[1].add_subplot(2, 1, 2)]
axes[0].set_ylim([-0.25, 5])
axes[1].set_ylim([-0.25, 5])
lines = []


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


FIRST_DRAW = True
line_types = ['b.', 'r.']


def update(deviations):
    global lines
    global FIRST_DRAW
    if FIRST_DRAW:
        FIRST_DRAW = False
        for i in range(2):
           lines.append(axes[i].plot(deviations[i][0], deviations[i][3], line_types[i], linewidth=0.1))
    else:
        for i in range(2):
           lines[i][0].set_xdata(deviations[i][0])
           lines[i][0].set_ydata(deviations[i][3])
        fig.canvas.draw()
        fig.canvas.flush_events()
        # pl.draw()
        # pl.clf()
    pl.pause(0.001)


def add_statistics(deviations, statistics, model_No):
    for i in range(2):
        for j in range(len(deviations[i][0])):
            statistics.append(
                [str(model_No), str(i), str(deviations[i][0][j]), str(deviations[i][1][j]), str(deviations[i][2][j]),
                 str(deviations[i][3][j]), str(deviations[i][4][j])])


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
    deviations = [[[], [], [], [], []], [[], [], [], [], []]]
    for step, (x, y, gesture_types, users, sample_ids) in enumerate(provider()):
        fetches = [model.cost, model.predict_op, eval_op]
        if status == 'test':
            fetches.append(model.logits)
            fetches.append(model.costs)
        feed_dict = {}
        feed_dict[model.acc_input_data] = x[0]
        feed_dict[model.emg_input_data] = x[1]
        feed_dict[model.targets] = y
        if status == 'test':
            cost, predict_val, _, logits, cost_list = session.run(fetches, feed_dict)
            # print(y, logits)
            for i in range(len(logits)):
                if y[i][0] == 0:
                    deviations[1][0].append(gesture_types[i])
                    deviations[1][1].append(users[i])
                    deviations[1][2].append(sample_ids[i])
                    deviations[1][3].append(cost_list[i])
                    deviations[1][4].append(predict_val[i])
                else:
                    deviations[0][0].append(gesture_types[i])
                    deviations[0][1].append(users[i])
                    deviations[0][2].append(sample_ids[i])
                    deviations[0][3].append(cost_list[i])
                    deviations[0][4].append(predict_val[i])
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
            # print("%.3f speed: %.0f wps time cost: %.3fs precision: %.3f debug_value: %.3f" %
            #       (step * 1.0 / epoch_size,
            #        sum / (time.time() - start_time), time.time() - stage_time, right_num / sum, debug_val / sum))
            stage_time = time.time()

    return np.exp(costs / iters), right_num / sum, debug_val / sum, deviations, sum / (time.time() - start_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_type', type=int, default=0)
    parser.add_argument('--model_dir', type=str, default="./model")
    parser.add_argument('--config_dir', type=str, default="./config.json")
    args = parser.parse_args()
    model_dir = args.model_dir
    train_type = args.train_type
    config_dir = args.config_dir
    provider = Provider(config_dir)
    provider.status = 'train'
    config = provider.get_config()
    eval_config = config.copy()
    eval_config['batch_size'] = 1
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    graph_dir = os.path.join(model_dir, "graphs")
    statistics_path = os.path.join(model_dir, "statistics.csv")
    statistics = []
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)

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
        with open(os.path.join(model_dir, "model_config.json"), 'w') as file_handle:
            file_handle.write(json.dumps(config))
        log = open(os.path.join(model_dir, "training_log.txt"), 'w')
        for i in range(config['max_epoch']):
            # lr_decay = config['lr_decay'] ** max(i - config['max_epoch'], 0.0)
            # m.assign_lr(session, config['learning_rate'] * lr_decay)
            m.assign_lr(session, config['learning_rate'])
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity, precision, debug_val, _, speed = run_epoch(session, m, provider, 'train', m.train_op,
                                                                         verbose=True)
            print("TRAIN COST:", train_perplexity, "TRAIN PRECISION: %.3f (%.3f)" % (precision, debug_val),
                  "SPEED: %.0f s" % speed)
            test_perplexity, precision, debug_val, deviations, _ = run_epoch(session, mtest, provider, 'test',
                                                                             tf.no_op())
            update(deviations)
            print("TEST COST:", test_perplexity, "TEST PRECISION: %.3f (%.3f)" % (precision, debug_val))
            if precision > best_precision or best_cost > test_perplexity:
                if precision > best_precision:
                    best_precision = precision
                if best_cost > test_perplexity:
                    best_cost = test_perplexity
                save_path = saver.save(session, '%s/model' % model_dir, global_step=model_No)
                log.write("No: %d   cost: %s, precision: %s, debug_val: %s\n" % (
                    model_No, test_perplexity, precision, debug_val))
                model_No += 1
                add_statistics(deviations, statistics, model_No)
                pl.savefig(os.path.join(graph_dir, "deviations_%d.png" % model_No), dpi=72)
                print("SAVE!!!!", "Model saved in file: %s" % save_path)
            print()
            # print("Ending Time:", datetime.now())
        print("BEST PRECISION", best_precision)
        with open(statistics_path, 'w') as file_handle:
            csv_str_list = ["models, is_tense, gesture_type, user_id, sample_id, cost, predict_val"]
            for statistic in statistics:
                csv_str = ",".join(statistic)
                csv_str_list.append(csv_str)
            csv_content = "\n".join(csv_str_list)
            file_handle.write(csv_content)
        log.close()

main()
