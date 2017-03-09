"""
A Dynamic Recurrent Neural Network (LSTM) implementation experiment using
TensorFlow library. This experiment is using a set of social data to classify its
sentiment. The sequences of words in each sentence have variable length.

Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Editor: Thanh L.X.

Reference: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf


from data.data_preprocess import one_hot_table
from itertools import izip

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("summaries_dir", "/tmp/lstm_for_SA_logs", "directory to summaries")
# flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("starter_learning_rate", 0.01, "starter learning rate")
flags.DEFINE_integer("training_iters", 100000, "number of training iterations")
flags.DEFINE_integer("batch_size", 100, "size of batch")
flags.DEFINE_integer("display_step", 10, "number of steps ==> display")
flags.DEFINE_integer("seq_max_len", 15, "sequence max length")
flags.DEFINE_integer("n_hidden", 100, "number of hidden neurons")
flags.DEFINE_integer("n_classes", 4, "number of sentiment classes - [0,1,2,3]")

flags.DEFINE_float("dropout", 0.5, "dropout - probability to keep units")

flags.DEFINE_integer("n_add", 4, "number of additional features")

# flags.DEFINE_integer("alpha", 0.1, "regularized term for balancing context and sentiment")


# ===================== #
# SENTIMENT DATA PARSER
# ===================== #
class SequenceData(object):
    # def __init__(self, n_samples=1000, max_seq_len=200, min_seq_len=1, data_in=None, label_in=None, table=None):
    def __init__(self, n_samples=1000, max_seq_len=200, min_seq_len=1, data_in=None, label_in=None, table=None,
                 add_features=None):
        self.data = []
        self.labels = []
        self.seqlen = []
        self.add = []
        if data_in and label_in:
            dim = table.shape[1]
            n_class = 4
            with open(data_in) as in_data, open(label_in) as in_label:
                for x, y in izip(in_data, in_label):
                    x = x.strip().split()
                    y = y.strip()
                    length = len(x)
                    assert length >= min_seq_len
                    zero = [0.] * dim
                    if length > max_seq_len:
                        x = x[:max_seq_len]
                        # s = [table[i] for i in range(max_seq_len)]
                        s = [table[int(i)] for i in x]
                        self.seqlen.append(max_seq_len)
                    else:
                        # s = [table[i] for i in range(length)]
                        s = [table[int(i)] for i in x]
                        s += [zero for _ in range(max_seq_len - length)]
                        self.seqlen.append(length)

                    self.data.append(s)
                    s_label = [0] * n_class
                    s_label[int(y) - 1] = 1
                    self.labels.append(s_label)

            # parsing additional features to sequence data
            with open(add_features) as features_add:
                for line in features_add:
                    line = line.strip().split()
                    add_feat = [5*float(i) for i in line]
                    self.add.append(add_feat)

        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])

        batch_add = (self.add[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])

        self.batch_id = min(self.batch_id + batch_size, len(self.data))

        # return batch_data, batch_labels, batch_seqlen
        return batch_data, batch_labels, batch_seqlen, batch_add

# ===================== #
#          MODEL
# ===================== #


# clear the default graph stack and reset the global default graph
print ("Reset the global default graph.. ")
tf.reset_default_graph()

# Parameters

#learning_rate = 0.01
#training_iters = 1000000
#batch_size = 100
#display_step = 10

# Network Parameters

#seq_max_len = 20  # Sequence max length
#n_hidden = 100  # hidden layer num of features
#n_classes = 4  # number of sentiment classes - [0,1,2,3]

# Specify path-to-source
source_path = "./data/"

length_dict = 6988
length_class = 7

Table = one_hot_table(source_path + "Dictionary_list_fixed", length_dict, length_class)

print("===================")
print("Dynamic RNN for SA")
print("===================")
print("Loading lookup table")
print(Table)
print(Table.shape)


# Passing data
print("Passing data")
# trainset = SequenceData(n_samples=5000, max_seq_len=FLAGS.seq_max_len, data_in=source_path + "train_data_SA",
#                        label_in=source_path + "train_label_SA", table=Table)

# training data with additional features
trainset = SequenceData(n_samples=5000, max_seq_len=FLAGS.seq_max_len, data_in=source_path + "train_data_SA",
                        label_in=source_path + "train_label_SA", table=Table, add_features=source_path + "train_add_SA")


# testset = SequenceData(n_samples=694, max_seq_len=FLAGS.seq_max_len, data_in=source_path + "test_data_SA",
#                       label_in=source_path + "test_label_SA", table=Table)

# test data with additional features
testset = SequenceData(n_samples=694, max_seq_len=FLAGS.seq_max_len, data_in=source_path + "test_data_SA",
                       label_in=source_path + "test_label_SA", table=Table, add_features=source_path + "test_add_SA")

print("check the data size")
print(len(trainset.data), len(trainset.data[0]), len(trainset.data[0][0]))
print(len(testset.data), len(testset.data[0]), len(testset.data[0][0]))


# tf Graph input
print("check input dimension")
dimension = Table.shape[1]
print(dimension)

print("start training ..")

# Placeholder for data
x = tf.placeholder("float", [None, FLAGS.seq_max_len, dimension])
# Placeholder for label
y = tf.placeholder("float", [None, FLAGS.n_classes])
# A placeholder for indicating each sequence length
seq_len = tf.placeholder(tf.int32, [None])

# Placeholder for dropout
keep_prob = tf.placeholder(tf.float32)

# Placeholder
addition = tf.placeholder("float", [None, FLAGS.n_add])


# Define weights
with tf.name_scope("weights"):
    weights = {
        # 'out': tf.Variable(tf.random_normal([FLAGS.n_hidden, FLAGS.n_classes]))
        'out': tf.Variable(tf.random_normal([FLAGS.n_hidden + FLAGS.n_add, FLAGS.n_classes]))
    }
with tf.name_scope("biases"):
    biases = {
        'out': tf.Variable(tf.random_normal([FLAGS.n_classes]))
    }


def dynamic_rnn(x, seqlen, weights, biases, dropout, addition):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, dimension])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, FLAGS.seq_max_len, x)

    # Define a lstm cell with tensorflow
    # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.pack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Generate corrupted text

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * FLAGS.seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, FLAGS.n_hidden]), index)

    outputs = tf.concat(1, [outputs, addition])


    # dropout before softmax layer
    outputs = tf.nn.dropout(outputs, dropout)
    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']



pred = dynamic_rnn(x, seq_len, weights, biases, keep_prob, addition)



# Define loss and optimizer
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
tf.scalar_summary("cost", cost)

with tf.name_scope("learning_rate"):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.starter_learning_rate, global_step, 10, 0.98, staircase=True)
    # learning_rate = FLAGS.starter_learning_rate
tf.scalar_summary("learning_rate", learning_rate)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.scalar_summary("accuracy", accuracy)

# Merge all the summaries and write them out to /tmp/lstm_for_SA_logs (by default)
if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
tf.gfile.MakeDirs(FLAGS.summaries_dir)

merged = tf.merge_all_summaries()
sess = tf.InteractiveSession()
train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                        sess.graph)
test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * FLAGS.batch_size < FLAGS.training_iters:
        # batch_x, batch_y, batch_seqlen = trainset.next(FLAGS.batch_size)
        batch_x, batch_y, batch_seqlen, batch_add = trainset.next(FLAGS.batch_size)
        # print (batch_add)
        # Run optimization (back-prop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seq_len: batch_seqlen, keep_prob: FLAGS.dropout, addition: batch_add})
        if step % FLAGS.display_step == 0:
            # Calculate batch accuracy
            # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
            #                                    seq_len: batch_seqlen})
            summary, acc = sess.run([merged, accuracy], feed_dict={x: batch_x, y: batch_y, seq_len: batch_seqlen,
                                                                   keep_prob: 1., addition: batch_add})
            train_writer.add_summary(summary, step)

            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seq_len: batch_seqlen, keep_prob: 1., addition: batch_add})

            print("Iter " + str(step * FLAGS.batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seq_len = testset.seqlen
    test_add = testset.add
    summary, acc = sess.run([merged, accuracy], feed_dict={x: test_data, y: test_label, seq_len: test_seq_len,
                                                           keep_prob: 1., addition: test_add})
    test_writer.add_summary(summary, step)
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                        seq_len: test_seq_len, keep_prob: 1., addition: test_add}))
