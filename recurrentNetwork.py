import tensorflow as tf
import numpy as np


def create_model(emb_size, vocab_size, lstm_hidden_size, T, learning_rate=0.001):
    """Create a LSTM neural network"""

    pad_vector = tf.zeros(shape=(1, emb_size), dtype=tf.float32, name="zero_padding")
    symbol_embedding = tf.get_variable('symbol_embeddings', shape=(vocab_size, emb_size), dtype=tf.float32)

    symbol_embedding = tf.concat([pad_vector, symbol_embedding], axis=0)

    input_ = tf.placeholder(shape=[None, T], dtype=tf.int32)
    labels_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    embedded = tf.nn.embedding_lookup(symbol_embedding, input_)

    lstm = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)
    outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=embedded, dtype=tf.float32)
    output = outputs[:, -1, :]
    logits = tf.keras.layers.Dense(1)(output)

    classify = tf.nn.sigmoid(logits)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_), axis=0)

    train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

    return {
        'train': train,
        'input': input_,
        'labels': labels_,
        'loss': loss,
        'classify': classify
    }


def evaluate(tf_session, tf_loss, tf_classify, data, labels):
    """
    Evaluate loss and accuracy on a single minibatch
    :param tf_session: current opened session
    :param tf_loss: tensor for calculating loss
    :param tf_classify: tensor for calculating sigmoid activations
    :param data: data from the current batch
    :param labels: labels from the current batch
    :return: loss_value, accuracy_value
    """

    loss_val, predict = tf_session.run([tf_loss, tf_classify], {
        input_: data,
        labels_: labels
    })
    acc_val = accuracy_score(labels, np.where(predict > 0.5, 1, 0))

    return loss_val, acc_val
