# -*- coding: utf-8 -*-
# @Time : 2019/6/25
# @Author : mch

import os
import re
import sys
import getopt
import tensorflow as tf
import pandas as pd

from itertools import chain


def main(argv):
    data_dir = ""
    task = ""
    help_short = "preprocessor.py -t <task> -d <data_dir>"
    help_long = "preprocessor.py --task=<task> --datadir=<data_dir>"
    if not len(argv) == 4:
        print('Error: params count error, need 4 params, but %d was given' % len(argv))
        print('Error:' + help_short)
        print('   or:' + help_long)
        sys.exit(1)
    try:
        # options, args = getopt.getopt(args, shortopts, longopts=[])

        # 参数args：一般是sys.argv[1:]。过滤掉sys.argv[0]，它是执行脚本的名字，不算做命令行参数。
        # 参数shortopts：短格式分析串。例如："hp:i:"，h后面没有冒号，表示后面不带参数；p和i后面带有冒号，表示后面带参数。
        # 参数longopts：长格式分析串列表。例如：["help", "ip=", "port="]，help后面没有等号，表示后面不带参数；ip和port后面带冒号，表示后面带参数。

        # 返回值options是以元组为元素的列表，每个元组的形式为：(选项串, 附加参数)，如：('-i', '192.168.0.1')
        # 返回值args是个列表，其中的元素是那些不含'-'或'--'的参数。
        opts, args = getopt.getopt(argv, "ht:d:", ["help", "task=", "datadir="])
    except getopt.GetoptError:
        print('Error:' + help_short)
        print('   or:' + help_long)
        sys.exit(2)
    # 处理 返回值options是以元组为元素的列表。
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_short)
            print(help_long)
            sys.exit()
        elif opt in ("-t", "--task"):
            task = arg
        elif opt in ("-d", "--datadir"):
            data_dir = arg
            print('data_dir：', data_dir)
            if not os.path.exists(data_dir):
                print('Error: ' + data_dir + " does not exist, please check your params")
                sys.exit(3)
    if task == 'cws':
        cws(data_dir)
    elif task == 'ner':
        ner(data_dir)
    return


def cws(data_dir):
    dfs = []
    data_files = os.listdir(data_dir)
    for file in data_files:
        if not os.path.isdir(file):
            df = generate_dataframe(data_dir + '\\' + file, ner=False)
            dfs.append((data_dir + '\\' + file, df))
    labels = ['x', 's', 'b', 'm', 'e']
    word2id, label2id = create_dict(dfs, data_dir, labels)
    create_tfrecords(word2id, label2id, dfs, 32)
    return


def ner(data_dir):
    dfs = []
    data_files = os.listdir(data_dir)
    for file in data_files:
        if not os.path.isdir(file):
            df = generate_dataframe(data_dir + '\\' + file, ner=True)
            dfs.append((data_dir + '\\' + file, df))
    labels = ['N', 'XX', 'XXX', 'S-sym', 'B-sym', 'M-sym', 'E-sym', 'S-bod', 'B-bod', 'M-bod', 'E-bod', 'S-tre',
              'B-tre', 'M-tre', 'E-tre', 'S-dis', 'B-dis', 'M-dis', 'E-dis', 'S-tes', 'B-tes', 'M-tes', 'E-tes', 'B-nt',
              'M-nt', 'E-nt']
    word2id, label2id = create_dict(dfs, data_dir, labels)
    dfs_no_TBD = [(j, i[~i['label'].astype(str).str.contains('TBD')])
                  for j, i in dfs]
    create_tfrecords(word2id, label2id, dfs_no_TBD, 32)
    return


def generate_dataframe(file, ner):
    '''
    create a csv file which contains words, label and length from file
    :param file: str path to file
    :return: dataframe contians words, label, length
    '''
    dataframe_file = file + "_data_label.csv"
    punctuation_pattern = r'[，。？！；：,?!;:]'
    text = []
    label = []
    if os.path.exists(dataframe_file):
        os.remove(dataframe_file)
    with open(file, "r", encoding='utf-8') as raw:
        for line in raw.readlines():
            sentences = re.split(punctuation_pattern, line.strip())
            sentences = list(filter(None, sentences))
            if ner:
                result = list(map(get_ner_labels_of_sentence, sentences))
                sentence_text = [i[0] for i in result]
                sentence_label = [i[1] for i in result]
            else:
                sentence_label = list(map(get_cws_labels_of_sentence, sentences))
                sentence_text = list(map(lambda x: x.replace(" ", ""), sentences))
                sentence_text = [list(i) for i in sentence_text]
            text.extend(sentence_text)
            label.extend(sentence_label)

    df = pd.DataFrame({'words': text, 'label': label}, index=range(len(text)))
    df['length'] = df['words'].apply(lambda words: len(words))
    df.to_csv(dataframe_file, index=False, sep=',', encoding='utf-8')
    print('generated dataframe file with data and label to: ' + dataframe_file)
    return df


def get_ner_labels_of_sentence(sentence):
    label_list = ['tes', 'bod', 'sym', 'tre', 'dis', 'nt']
    sentence = sentence.strip().replace(' ', '')
    stack = [None for i in sentence]
    words = []
    labels = []
    temp_list = []
    add_to_stack = False
    layer = 0
    top = -1
    i = 0
    index = -1
    multi = 0
    while i < len(sentence):
        if sentence[i] == '[':
            top = top + 1
            stack[top] = '['
            add_to_stack = True
            i = i + 1
        elif sentence[i] == ']':
            if sentence[i + 1:i + 4] in label_list:
                label = sentence[i + 1:i + 4]
                i = i + 4
            elif sentence[i + 1: i + 3] in label_list:
                label = sentence[i + 1:i + 3]
                i = i + 3
            else:
                while stack[top] != '[' and top > -1:
                    top -= 1
                top -= 1
                temp_list = []
                i = i + 1
                if top == -1:
                    add_to_stack = False
                layer = 0
                continue
            while stack[top] != '[' and top > -1:
                temp_list.append(stack[top])
                top -= 1
            j = top
            top -= 1
            layer = 0
            while j > -1 and stack[j] == '[':
                layer = layer + 1
                j = j - 1
            temp_list.reverse()
            for k in range(len(temp_list)):
                if len(temp_list) == 1:
                    labels[temp_list[k]] = 'S-' + label
                else:
                    if k == 0:
                        if layer > 1:
                            labels[temp_list[k]] = ('X' * layer)
                            # multi = temp_list[k]
                        elif labels[temp_list[k] + 1] != 'TBD':
                            labels[temp_list[k]] = 'M-' + label
                        else:
                            labels[temp_list[k]] = 'B-' + label
                    elif k == len(temp_list) - 1:
                        labels[temp_list[k]] = 'E-' + label
                    else:
                        labels[temp_list[k]] = 'M-' + label
                    # words.append(temp_list[k])
            if top == -1:
                add_to_stack = False
            temp_list = []
            layer = 0
        elif add_to_stack:
            top = top + 1
            index = index + 1
            labels.append('TBD')
            stack[top] = index
            words.append(sentence[i])
            i = i + 1
        else:
            words.append(sentence[i])
            labels.append('N')
            index += 1
            i = i + 1
    return words, labels


def get_cws_labels_of_sentence(s):
    '''
    return a list of sentence's labels
    :param s: str
    :return: list(str)
    '''
    label = []
    for word in s.split():
        if len(word) == 1:
            label.extend(['s'])
        elif len(word) > 1:
            for i in range(len(word)):
                if i == 0:
                    label.extend(['b'])
                elif i == len(word) - 1:
                    label.extend(['e'])
                else:
                    label.extend(['m'])
    return label


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_tfrecords(word2id, label2id, dfs, max_len):
    '''

    :param word2id: pd.Series, first column word , second column id
    :param label2id: pd.Series, first colum label, second column id
    :param dfs: list[(file, dataframe)]
    :return: create tfrecords of train and validation
    '''
    for file, df in dfs:
        path = file + '.tfrecords'
        if os.path.exists(path):
            os.remove(path)
        writer = tf.python_io.TFRecordWriter(path)
        df['X'] = df['words'].apply(X_padding, args=(word2id, max_len))
        df['y'] = df['label'].apply(y_padding, args=(label2id, max_len))
        datas = list(df['X'])
        labels = list(df['y'])
        for i in range(len(datas)):
            example = tf.train.Example(features=tf.train.Features(feature=
                                                                  {'label': _int64_feature(labels[i]),
                                                                   'data': _int64_feature(datas[i])}))
            writer.write(example.SerializeToString())
        writer.close()
        print('created ' + path)


def X_padding(words, word2id, max_len):
    """

    :param words:
    :param word2id:
    :param max_len:
    :return:
    """
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))  # 短则补全
    return ids


def y_padding(label, label2id, max_len):
    """

    :param label:
    :param label2id:
    :param max_len:
    :return:
    """
    ids = list(label2id[label])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))  # 短则补全
    return ids


def create_dict(l, data_dir, labels):
    '''
    calculate and print vocabulary size and create word2id id2word label2id id2label files from l
    :param l: list((file, dataframe))
    :return: pd.series, first one is word2id, second one is label2id
    '''
    df_list = [i[1] for i in l]
    df_all = pd.concat(df_list, ignore_index=True)
    all_words = list(chain(*df_all['words'].values))
    word_series = pd.Series(all_words)
    word_series = word_series.value_counts()
    word_set = word_series.index
    word_set_ids = range(1, len(word_set) + 1)
    label_ids = range(len(labels))
    print("vocabulary size: " + str(len(word_set)))

    word2id = pd.Series(word_set_ids, index=word_set)
    id2word = pd.Series(word_set, index=word_set_ids)
    label2id = pd.Series(label_ids, index=labels)
    id2label = pd.Series(labels, index=label_ids)

    word2id.to_csv(data_dir + '/word2id.csv', encoding='utf-8')
    id2word.to_csv(data_dir + '/id2word.csv', encoding='utf-8')
    label2id.to_csv(data_dir + '/label2id.csv', encoding='utf-8')
    id2label.to_csv(data_dir + '/id2label.csv', encoding='utf-8')
    print('created word2id id2word label2id id2label')
    return word2id, label2id


def parse_sample(data_record):
    '''

    :param data_record:
    :return:
    '''
    features = {
        'label': tf.FixedLenFeature([32], tf.int64),
        'data': tf.FixedLenFeature([32], tf.int64)
    }
    sample = tf.parse_single_example(data_record, features)
    return sample


# 读取tfrecords文件
def read_tfrecords(filename, epoch, batch_size, shuffle_size):
    '''

    :param filename:
    :param epoch:
    :param batch_size:
    :param shuffle_size:
    :return:
    '''
    # 使用dataset模块读取数据
    datasets = tf.data.TFRecordDataset(filenames=[filename])
    # 对每一条record进行解析
    dataset = datasets.map(parse_sample)
    batch_set = dataset.shuffle(shuffle_size).repeat(epoch).batch(batch_size=batch_size)
    # iterator = batch.make_one_shot_iterator()
    return batch_set


if __name__ == '__main__':
    main(sys.argv[1:])
