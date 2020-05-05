from django.shortcuts import render
from web.Utils.WordVecs import *

from keras.preprocessing.sequence import pad_sequences
from sklearn import decomposition

import json
import os
import collections
import tensorflow as tf
import numpy as np


def min_max_range(x, range_values):
    return [round( ((xx - min(x)) / (1.0*(max(x) - min(x)))) * (range_values[1] - range_values[0]) + range_values[0], 2) for xx in x]


def idx_sent(sent, w2idx):
    idx = np.zeros((1, 39))
    i = 0
    for w in sent:
        if w in w2idx.keys():
            idx[0][i] = w2idx[w]
        else:
            idx[0][i] = 0
        i += 1
    return np.array(idx)


def add_unknown_words(vecs, wordvecs, vocab, min_df=1, dim=50):
    """
    For words that occur at least min_df times, create a separate word vector.
    0.25 is chosen so the unk vectors have approximately the same variance as pretrained ones
    """
    num_word_unknown = 0.0
    num_word_unknown_lower_cased = 0.0
    for word in vocab:
        if word not in wordvecs:
            num_word_unknown += 1
            if word.lower() not in wordvecs:
                num_word_unknown_lower_cased += 1

        if word not in wordvecs and vocab[word] >= min_df:
            wordvecs[word] = np.random.uniform(-0.25, 0.25, dim)
    return wordvecs


def split_sent(sent):
    for char in '!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~':
        sent = sent.replace('{}'.format(char), ' {} '.format(char))
    return sent.split()


def preprocess_words(words):
    vecs = WordVecs('./static/embedding/google.txt', 'word2vec')
    dim = vecs.vector_size
    # max_length = len(words)

    vocab = collections.defaultdict(int)
    for w in words:
        vocab[w] += 1
    # create a dict of words that are in our word2vec embeddings
    # wordvecs: String -> embedding_vec
    wordvecs = {}
    for w in vecs._w2idx.keys():
        if w in vocab:
            wordvecs[w] = vecs[w]
    # Assign random w2v vectors to the unknown words. These are random uniformly distrubuted vectors of size dim.
    wordvecs = add_unknown_words(vecs, wordvecs, vocab, min_df=1, dim=dim)

    f = open('./static/model/word_idx_map.json', 'r')
    word_idx_map = json.load(f)
    X = idx_sent(wordvecs, word_idx_map)
    pad_sequences(sequences=X, maxlen=39, padding="post")
    #print(X)

    return X


def predict(X, model_path, path, length):
    sess = tf.Session()

    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, tf.train.latest_checkpoint(path))
    graph = tf.get_default_graph()

    # get pretrained parameters
    emb = np.load('./static/model/W.npy')
    # emb = sess.graph.get_tensor_by_name("embedding_table:0").eval(session=sess)
    input_sentence = graph.get_operation_by_name('input_sentence').outputs[0]
    dropout_keep_probability = graph.get_operation_by_name('dropout_keep_probability').outputs[0]

    feed_dict = {input_sentence.name: X,
                 # targets.name: Y,
                 'embedding_table_1:0': emb,
                 dropout_keep_probability.name: 0.2}

    result = {}
    # get softmax weights in attention layer
    weights = sess.run("Stack-Layer-1/attention_weights:0", feed_dict)
    weights = weights.reshape(weights.shape[2], weights.shape[3])
    weights = weights[:length]
    # print(weights.shape)

    # get input_embedding
    input_embedding = sess.run("Input_Embeddings_Dropout/dropout/mul_1:0", feed_dict)
    input_embedding = input_embedding.reshape(input_embedding.shape[1], input_embedding.shape[2])
    input_embedding = input_embedding[:length]

    # get q, k, v
    q = sess.run("Stack-Layer-1/dense/Relu:0", feed_dict)
    q = q.reshape(q.shape[1], q.shape[2])
    q = q[:length]
    k = sess.run("Stack-Layer-1/dense_1/Relu:0", feed_dict)
    k = k.reshape(k.shape[1], k.shape[2])
    k = k[:length]
    v = sess.run("Stack-Layer-1/dense_2/Relu:0", feed_dict)
    v = v.reshape(v.shape[1], v.shape[2])
    v = v[:length]

    #get mean of word embedding
    word_embedding = sess.run("Stack-Layer-1/Reshape_7:0", feed_dict)
    word_embedding = word_embedding.reshape(word_embedding.shape[1], word_embedding.shape[2])
    word_embedding = word_embedding[:length]

    # get wq, wk, wv
    wq = sess.graph.get_tensor_by_name("Stack-Layer-1/dense/kernel:0").eval(session=sess)
    result["wq"] = wq
    result["input_embedding"] = input_embedding

    result["softmax_weights"] = weights
    result["word_embedding"] = word_embedding
    result["q"] = q
    result["k"] = k
    result['v'] = v

    return result


# def new_qkv_format(result, words, max_=1):
#     q = result['q']
#     q_data = {}
#     dates = [j for j in range(30)]
#     q_data['dates'] = dates
#     series = []
#     for i in range(q.shape[0]):
#         row = []
#         cand = []
#         for j in range(q.shape[1]):
#             cand.append(q[i][j])
#             if len(cand) == max_:
#                 row.append(str(np.max(np.array(cand))))
#                 cand = []
#         # print("row {}".format(row))
#         # print("q[i] {}".format(q[i].tolist()))
#         tmp = {}
#         tmp['name'] = words[i]
#         # tmp['values'] = q[i].tolist()
#         tmp['values'] = row
#         series.append(tmp)
#     q_data['series'] = series
#
#     return q_data

def word_embedding_format(word_embedding, words):

    return 

def qkv_format(result, words, max_=1, mat='q'):
    matrix = result[mat]
    dict = []
    for i in range(matrix.shape[0]):
        matrix[i] = min_max_range(matrix[i], (-0.5, 0.5))
        cand = []
        tmp = {}
        tmp['row'] = words[i]
        val = 0
        for j in range(matrix.shape[1]):
            cand.append(matrix[i][j])
            if len(cand) == max_:
                tmp['value'] = str(val / matrix.shape[1])
                tmp['color'] = str(np.max(np.array(cand)))
                dict.append(tmp)
                cand = []
                tmp = {}
                tmp['row'] = words[i]
                val += 1
    return dict


def qkv_format_mean(result, words, mean_=1, mat='q'):
    matrix = result[mat]
    dict = []
    for i in range(matrix.shape[0]):
        matrix[i] = min_max_range(matrix[i], (-0.5, 0.5))
        cand = []
        tmp = {}
        tmp['row'] = words[i]
        val = 0
        for j in range(matrix.shape[1]):
            cand.append(matrix[i][j])
            if len(cand) == mean_:
                tmp['value'] = str(val / matrix.shape[1])
                tmp['color'] = str(np.mean(np.array(cand)))
                dict.append(tmp)
                cand = []
                tmp = {}
                tmp['row'] = words[i]
                val += 1
    return dict


# def input_embedding_format(result, words):
#     input_embedding = result["input_embedding"]
#     input_embedding_dict = []
#     for i in range(input_embedding.shape[0]):
#         # input_embedding[i] = min_max_range(input_embedding[i], (-0.5, 0.5))
#         for j in range(input_embedding.shape[1]):
#             tmp = {}
#             tmp['row'] = words[i]
#             tmp['value'] = str(j / input_embedding.shape[1])
#             # tmp['color'] = str(-1 + (1 + 1)/(np.max(q)-np.min(q))*(q[i][j]-np.min(q)))
#             tmp['color'] = str(input_embedding[i][j])
#             # tmp['color'] = str(1.0/(1.0 + np.exp(-q[i][j])))
#             input_embedding_dict.append(tmp)
#
#     return input_embedding_dict


# def wqwkwv_format(result):
#     wq = result["wq"]
#     wq_dict = {}
#
#     wq_dict['values'] = wq.tolist()
#     names = [i for i in range(wq.shape[0])]
#     years = [j for j in range(wq.shape[1])]
#     year = int(wq.shape[1]/2)
#
#     # wq_dict["values"] = values
#     wq_dict["names"] = names
#     wq_dict["years"] = years
#     wq_dict["year"] = year
#
#     return wq_dict


def multi_softmax_format(result, words):
    softmax_weights = result['softmax_weights']
    softmax_weights_data = {}
    dates = [j for j in range(len(softmax_weights))]
    softmax_weights_data['words'] = words
    series = []
    for i in range(softmax_weights.shape[0]):
        row = []
        for j in range(softmax_weights.shape[1]):
            row.append(str(softmax_weights[i][j]))
        tmp = {}
        tmp['name'] = words[i]
        tmp['values'] = row
        series.append(tmp)
    softmax_weights_data['series'] = series
    softmax_weights_data['y'] = "Softmax Value"
    softmax_weights_data['shunxu'] = [i for i in range(len(words))]
    return softmax_weights_data

def softmax_matrix_format(result, words):
    softmax_weights = result['softmax_weights']
    softmax_matrix = {}
    series = []
    n = softmax_weights.shape[0]
    for i in range(n):
        for j in range(n):
            tmp = {}
            tmp['L1'] = words[i]
            tmp['L2'] = words[j]
            tmp['value'] = str(round(softmax_weights[i][j],4))
            series.append(tmp)
    softmax_matrix['table'] = series
    return softmax_matrix

def word_cloud_format(result, words):
    word_embedding = result["word_embedding"]
    num_col = word_embedding.shape[1]
    num_row = word_embedding.shape[0]
    word_freq = [0]*num_row
    for i in range(num_col):
        avg = np.sum(word_embedding[:, i])/num_row
        sub = [abs(word_embedding[j][i]-avg) for j in range(num_row)]
        minpos = sub.index(min(sub))
        word_freq[minpos] = word_freq[minpos] + 1
    series = []
    for i in range(num_row):
        tmp = {}
        tmp['name'] = words[i]
        tmp['value'] = word_freq[i]
        series.append(tmp)
    word_cloud = {}
    word_cloud['series'] = series
    print(series)
    return word_cloud

def q(request):
    path = './static/model'
    sent = "I don't like the taste of this restaurant"
    try:
        files = os.listdir(path)
        model_path = ""
        for file in files:
            if os.path.splitext(file)[1] == '.meta':
                model_path = path + '/' + file
    except:
        context = {}
        context['test'] = "Can't find model!"
        return render(request, 'error.html', {'context': json.dumps(context)})

    words = split_sent(sent)
    length = len(words)

    X = preprocess_words(words)

    result = predict(X, model_path, path, length)
    dict = qkv_format(result, words, mat='q')
    dict_max = qkv_format(result, words, max_=5, mat='q')
    dict_mean = qkv_format_mean(result, words, mean_=5, mat='q')

    #horiz

    context = {}
    context['dict'] = dict
    context['dict_max'] = dict_max
    context['dict_mean'] = dict_mean

    return render(request, 'index.html', {'context': json.dumps(context),
                                          'matrix_name': 'Q'})


def k(request):

    path = './static/model'
    sent = "I don't like the taste of this restaurant"
    try:
        files = os.listdir(path)
        model_path = ""
        for file in files:
            if os.path.splitext(file)[1] == '.meta':
                model_path = path + '/' + file
    except:
        context = {}
        context['test'] = "Can't find model!"
        return render(request, 'error.html', {'context': json.dumps(context)})

    words = split_sent(sent)
    length = len(words)

    X = preprocess_words(words)

    result = predict(X, model_path, path, length)
    dict = qkv_format(result, words, mat='k')
    dict_max = qkv_format(result, words, max_=5, mat='k')
    dict_mean = qkv_format_mean(result, words, mean_=5, mat='k')


    context = {}
    context['dict'] = dict
    context['dict_max'] = dict_max
    context['dict_mean'] = dict_mean

    return render(request, 'index.html', {'context': json.dumps(context),
                                          'matrix_name': 'K'})


def v(request):
    matrix_name = 'v'
    path = './static/model'
    sent = "I don't like the taste of this restaurant"
    try:
        files = os.listdir(path)
        model_path = ""
        for file in files:
            if os.path.splitext(file)[1] == '.meta':
                model_path = path + '/' + file
    except:
        context = {}
        context['test'] = "Can't find model!"
        return render(request, 'error.html', {'context': json.dumps(context)})

    words = split_sent(sent)
    length = len(words)

    X = preprocess_words(words)

    result = predict(X, model_path, path, length)
    dict = qkv_format(result, words, mat=matrix_name)
    dict_max = qkv_format(result, words, max_=5, mat=matrix_name)
    dict_mean = qkv_format_mean(result, words, mean_=5, mat=matrix_name)


    context = {}
    context['dict'] = dict
    context['dict_max'] = dict_max
    context['dict_mean'] = dict_mean

    return render(request, 'index.html', {'context': json.dumps(context),
                                          'matrix_name': 'V'})


def softmax(request):
    path = './static/model'
    sent = "I like the taste of this restaurant"
    try:
        files = os.listdir(path)
        model_path = ""
        for file in files:
            if os.path.splitext(file)[1] == '.meta':
                model_path = path + '/' + file
    except:
        context = {}
        context['test'] = "Can't find model!"
        return render(request, 'error.html', {'context': json.dumps(context)})

    words = split_sent(sent)
    length = len(words)

    X = preprocess_words(words)
    result = predict(X, model_path, path, length)
    multi_softmax = multi_softmax_format(result, words)
    softmax_matrix = softmax_matrix_format(result, words)
    word_cloud = word_cloud_format(result, words)

    context = {}
    context['multi_softmax'] = multi_softmax
    context['softmax_matrix'] = softmax_matrix
    context['word_cloud'] = word_cloud

    return render(request, 'words.html', {'context': json.dumps(context)})


def search(request):
    sent = request.GET.get('q')

    context = {}
    context['test'] = 'hello'

    return render(request, 'index.html', {'context': json.dumps(context)})



# def max_(request):
#     # sent = request.GET.get('q')
#     path = './static/model'
#     sent = "I don't like the taste of this restaurant"
#     try:
#         files = os.listdir(path)
#         model_path = ""
#         for file in files:
#             if os.path.splitext(file)[1] == '.meta':
#                 model_path = path + '/' + file
#     except:
#         context = {}
#         context['test'] = "Can't find model!"
#         return render(request, 'error.html', {'context': json.dumps(context)})
#
#     words = split_sent(sent)
#     length = len(words)
#
#     X = preprocess_words(words)
#
#     result = predict(X, model_path, path, length)
#     print("input_embedding {}".format(result["input_embedding"].shape))
#     print("q {}".format(result["q"].shape))
#     input_embedding_dict = input_embedding_format(result, words)
#     q_dict, k_dict, v_dict = qkv_format(result, words, max_=5)
#     wq_dict = wqwkwv_format(result)
#
#     # nq = new_qkv_format(result, words, max_=5)
#
#     context = {}
#     context['q_dict'] = q_dict
#     context['k_dict'] = k_dict
#     context['v_dict'] = v_dict
#     context['wq_dict'] = wq_dict
#     context['input_embedding_dict'] = input_embedding_dict
#     # context['nq'] = nq
#     print(q_dict)
#
#     return render(request, 'index.html', {'context': json.dumps(context)})