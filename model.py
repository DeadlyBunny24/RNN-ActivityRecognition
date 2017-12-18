import tensorflow as tf
from tensorflow.python import debug as tf_debug
from dataset import _parse_function

batch_size = 20
hidden_size = 400
num_classes = 2
learning_rate = 0.01
padding_size = 300
num_epochs = 100
feature_size = 100

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

with tf.name_scope('accuracy'):
    mask_as_int = tf.cast(sequence_mask,tf.int64)
    predictions_labels = tf.argmax(logits,2)
    # real_labels = labels
    # correct_prediction = tf.equal(predictions_labels,real_labels)
    # real_correct_prediction = tf.cast(correct_prediction,tf.int64)*mask_as_int
    # accuracy = tf.reduce_sum(tf.cast(real_correct_prediction,tf.float64)) /\
    #             tf.reduce_sum(tf.cast(mask_as_int,tf.float64))

    # This line is meant to extract the local variables associated with the accuracy.
    # Accordingly, this allows us to reset them when one batch is completed
    pred_vars = tf.contrib.framework.get_variables('accuracy', collection=tf.GraphKeys.LOCAL_VARIABLES)

    _,accuracy = tf.metrics.accuracy(labels=labels,
                                     predictions=predictions_labels,
                                     weights=mask_as_int)

# Summaries collected for tensorboard.
with tf.name_scope('summaries'):
    acc_summary = tf.summary.scalar('acc',accuracy)
    loss_summary = tf.summary.scalar('loss',loss)
    hidden_summary = tf.summary.histogram('hidden_states',outputs)
    merge_during_train = tf.summary.merge([loss_summary,hidden_summary])

with tf.Session() as sess:
    global_step = 0
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    # The local initializer is necesary because of tf.metrics.accuracy
    sess.run([init_global,init_local])

    # Init of string handlers to move between the validation and training Datasets
    train_iterator_handle = sess.run(train_iterator.string_handle())
    test_iterator_handle = sess.run(test_iterator.string_handle())

    writer = tf.summary.FileWriter('./graphs/hs_{}/'.format(hidden_size), sess.graph)

    # This runs the debugger
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # This loop iterates through the training dataset.
    while True:
        try:
            global_step+=1
            _,summary_train = sess.run([train_step,merge_during_train],
                                        feed_dict={handle:train_iterator_handle})

            if global_step % 125 == 0:
                print 'Step: {}'.format(global_step)

                # This resets the local variables associatted with the acc counter
                sess.run(pred_vars)
                sess.run(test_iterator.initializer)

                # This loop iterates through the validation dataset.
                while True:
                    try:
                        summary_test = sess.run(acc_summary,
                                                feed_dict={handle:test_iterator_handle})
                    except tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError:
                        print 'End of validation dataset'
                        writer.add_summary(summary_train,global_step)
                        writer.add_summary(summary_test,global_step)
                        break

        except tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError:
            print 'End of training dataset'
            writer.close()
            break
