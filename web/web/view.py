from django.shortcuts import render
from web.Utils.WordVecs import *
from keras.preprocessing.sequence import pad_sequences

import json
import os
import collections
import tensorflow as tf


def idx_sent(sent, w2idx):
    idx = np.zeros((1, 23))
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
    print(X)

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

    # get q, k, v
    q = sess.run("Stack-Layer-1/dense/Relu:0", feed_dict)
    q = q.reshape(q.shape[1], q.shape[2])
    q = q[:length]

    # get wq, wk, wv
    wq = sess.graph.get_tensor_by_name("Stack-Layer-1/dense/kernel:0").eval(session=sess)
    result["wq"] = wq

    result["softmax_weights"] = weights
    result["q"] = q

    return result


def qkv_format(result):
    q = result["q"]

    q_dict = []
    # print(q[0][1])
    # print(q[1][1])
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            tmp = {}
            tmp['row'] = str(i)
            tmp['value'] = str(j / q.shape[1])
            # tmp['color'] = str(-1 + (1 + 1)/(np.max(q)-np.min(q))*(q[i][j]-np.min(q)))
            tmp['color'] = str(q[i][j])
            q_dict.append(tmp)

    return q_dict


def wqwkwv_format(result):
    wq = result["wq"]
    wq_dict = {}

    wq_dict['values'] = wq.tolist()
    # values = [[0]*wq.shape[1]]*wq.shape[0]
    # for i in range(wq.shape[0]):
    #     for j in range(wq.shape[1]):
    #         # values[i][j] = str(-1 + (1 + 1)/(np.max(wq)-np.min(wq))*(wq[i][j]-np.min(wq)))
    #         values[i][j] = str(wq[i][j])

    names = [i for i in range(wq.shape[0])]
    years = [j for j in range(wq.shape[1])]
    year = int(wq.shape[1]/2)

    # wq_dict["values"] = values
    wq_dict["names"] = names
    wq_dict["years"] = years
    wq_dict["year"] = year

    return wq_dict


def q(request):

    path = './static/model'
    sent = "Due to majorcontruction could not get any information or a tour . The building site looks great and currents reside traceaboutgneace"
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

    q_dict = qkv_format(result)
    wq_dict = wqwkwv_format(result)

    context = {}
    context['q_dict'] = q_dict
    context['wq_dict'] = wq_dict
    # print(q_dict)

    return render(request, 'index.html', {'context': json.dumps(context)})


def search(request):
    sent = request.GET.get('q')

    context = {}
    context['test'] = 'hello'

    return render(request, 'index.html', {'context': json.dumps(context)})