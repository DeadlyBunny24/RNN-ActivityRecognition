import tensorflow as tf
from tensorflow.python import debug as tf_debug

batch_size = 20
hidden_size = 300
num_classes = 2
learning_rate = 0.01
padding_size = 300
num_epochs = 100
feature_size = 100

def _parse_function(example_proto):
    features = {"seq": tf.FixedLenFeature([], tf.string, default_value=""),
              "label": tf.FixedLenFeature([], tf.int64, default_value=0),
              "seq_len": tf.FixedLenFeature([], tf.int64, default_value=0)}

    example = tf.parse_single_example(example_proto, features)
    array_features = tf.decode_raw(example['seq'],tf.float64)
    parsed_seq_len = example['seq_len']
    parsed_label = example['label']
    return array_features, parsed_label, parsed_seq_len

# This section uses the Dataset API to iterate through a tfRecords dataset.
# It iterates the dataset and applies tensor transformations to prepare the input
# -----------
with tf.name_scope('dataset'):
    filename = 'tfRecordFiles/train_orient2.tfrecords'
    dataset_train = tf.contrib.data.TFRecordDataset(filename)
    dataset_train = dataset_train.repeat(num_epochs)
    dataset_train = dataset_train.map(_parse_function)
    dataset_train = dataset_train.padded_batch(batch_size,
                                               padded_shapes=([padding_size*feature_size],[],[]))

    filename = 'tfRecordFiles/test_orient2.tfrecords'
    dataset_test = tf.contrib.data.TFRecordDataset(filename)
    dataset_test = dataset_test.map(_parse_function)
    dataset_test = dataset_test.padded_batch(batch_size,
                                             padded_shapes=([padding_size*feature_size],[],[]))

    train_iterator = dataset_train.make_one_shot_iterator()
    test_iterator = dataset_test.make_initializable_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(handle,
                                                           dataset_train.output_types,
                                                           dataset_train.output_shapes)

    next_element = iterator.get_next()

    input_reshape = tf.reshape(next_element[0],[batch_size,padding_size,feature_size])
    input_model = tf.transpose(input_reshape,[1,0,2],name='model_input')
    sequence_sizes = next_element[2]
    tile_labels = tf.tile(next_element[1],[padding_size])
    reshape_labels = tf.reshape(tile_labels,[padding_size,batch_size])
    labels = tf.transpose(reshape_labels,name='model_labels')
# --------------

# Creates a true false mask to prevent from training on padded elements
sequence_mask = tf.sequence_mask(sequence_sizes,padding_size,
                                dtype=tf.float64,
                                name='mask')

# TODO: Replace the tf.unstack call with a tf.while_loop.
input_model_ = tf.unstack(input_model,axis=0)

# The variable scope has an initializer parameter that allows to control the
# initilization of the variables inside of the scope.
with tf.variable_scope('RNN'):
    cell = tf.contrib.rnn.BasicRNNCell(hidden_size)

    state = tf.zeros([batch_size,hidden_size],dtype=tf.float64,name='state_init')

    outputs = []

    for batch in input_model_:
        output, state = cell(batch,state)
        outputs.append(output)

# ouputs has shape [sequence_length, batch_size, hidden_size]
# The transpose is necesary because 'loss' expects [bs,sl,hs] as 'logits'
outputs = tf.transpose(tf.stack(outputs),perm=[1,0,2],name='hidden_states')

# Output layer
logits = tf.contrib.layers.fully_connected(inputs=outputs,
                                           num_outputs=num_classes,
                                           activation_fn=tf.nn.sigmoid)

# Loss
loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                        targets=labels,
                                        weights=sequence_mask,
                                        name='loss')

# Optimizer
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    global_step = 0
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    # The local initializer is necesary because of tf.metrics.accuracy
    sess.run([init_global,init_local])

    # Init of string handlers to move between the validation and training Datasets
    train_iterator_handle = sess.run(train_iterator.string_handle())
    test_iterator_handle = sess.run(test_iterator.string_handle())

    # This loop iterates through the training dataset.
    while True:
        try:
            global_step+=1
            _, _loss = sess.run([train_step,loss],
                                        feed_dict={handle:train_iterator_handle})

            if global_step % 125 == 0:
                print 'Step: {}, Loss: {}'.format(global_step,_loss)

        except tf.errors.OutOfRangeError:
            print 'End of training dataset'
            break
