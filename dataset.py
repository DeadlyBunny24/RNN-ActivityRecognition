import numpy as np
import scipy.io
from collections import namedtuple
import os, re
import tensorflow as tf

subjects = [ 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
filter_subjects = list(set(range(1,41)).difference(subjects)) # create set difference for 40 subjects

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate_dataset(train,
                    cross_view,
                    num_classes=49,
                    include_orientation=True,
                    include_position=False,
                    include_color=False,
                    include_depth=False):

    # Train: Is used to determine wuether the dataset generated is for training or testing purposes
    #        in accordance with: "NTU RGB+D: Cross Subject Evaluation"
    # freq_scale: Factor that multiplies the nyquist frequency (factor*nyquist_frequency=factor*2*max_frequency)
    # num_steps: Number of states the RNN will use to truncate BPTT
    # include_x: Includes the info as part of the feature vector.
    # It is recommended that freq_scale == num_steps


    # select .mat files that are not mutual actions activities (punching, kicking, hugging, etc.) "S012C001P028R001A001"

    dir_name = 'skeleton_data_test'
    filenames = os.listdir(dir_name)
    filter_activity = re.compile(r'.+A0(?P<result>\d{2}).mat$')
    filter_test = re.compile(r'.+C001.+.mat$')
    if not cross_view:
        subjects_re = "|".join(['{num:02d}'.format(num=i) for i in filter_subjects])
        subjects_test = re.compile(r'.+P0({})+.+.mat$'.format(subjects_re))
    file_selection = []
    activity_label = []

    for filename in filenames:
        if not filename.startswith('S'): continue
        test = filter_test
        if not cross_view: test = subjects_test
        test_true = test.match(filename)
        if train:
            if test_true: continue
        else:
            if not test_true: continue
        file_num = int(filter_activity.match(filename).group('result'))
        if file_num <= num_classes: #num_classes=49 caps the activities to single skeleton ones
            activity_label.append(file_num-1)
            file_selection.append(filename)

    # List of ndarrays of shape (25,1031) that represent each action class

    all_activities_features = []
    label_seq = []
    seq_len = []

    # Open tfRecord file
    # TODO: Eliminate the os function
    if train:
        tfrecord_filename = os.path.join('tfRecordFiles', 'train_orient{}'.format(num_classes) + '.tfrecords')
    else:
        tfrecord_filename = os.path.join('tfRecordFiles', 'test_orient{}'.format(num_classes) + '.tfrecords')

    writer = tf.python_io.TFRecordWriter(tfrecord_filename)

    # process files to capture features
    for filename,label in zip(file_selection, activity_label):
        print ('filename:{}   label:{}'.format(filename,label))
        matlab_file_data = scipy.io.loadmat(dir_name + '/' + filename)
        skeletons = matlab_file_data['skeletons']
        frames = skeletons[0] # these are the captured frames for this action
        features = []
        for frame in frames:
            if frame['bodies'].shape == (1,0): continue

            # Range = [-10,10]
            feature_x = frame['bodies']['joints'][0][0]['x']
            feature_y = frame['bodies']['joints'][0][0]['y']
            feature_z = frame['bodies']['joints'][0][0]['z']

            # Range = [500,1000]
            feature_color_x = frame['bodies']['joints'][0][0]['colorX']
            feature_color_y = frame['bodies']['joints'][0][0]['colorY']

            # Range = [100,300]
            feature_depth_x = frame['bodies']['joints'][0][0]['depthX']
            feature_depth_y = frame['bodies']['joints'][0][0]['depthY']

            # Range = [-1,1]
            feature_ow = frame['bodies']['joints'][0][0]['orientationW']
            feature_ox = frame['bodies']['joints'][0][0]['orientationX']
            feature_oy = frame['bodies']['joints'][0][0]['orientationY']
            feature_oz = frame['bodies']['joints'][0][0]['orientationZ']

            f_x = [i[0][0] for i in feature_x.flat]
            f_y = [i[0][0] for i in feature_y.flat]
            f_z = [i[0][0] for i in feature_z.flat]
            f_c_x = [i[0][0] for i in feature_color_x.flat]
            f_c_y = [i[0][0] for i in feature_color_y.flat]
            f_d_x = [i[0][0] for i in feature_depth_x.flat]
            f_d_y = [i[0][0] for i in feature_depth_y.flat]
            f_ow = [i[0][0] for i in feature_ow.flat]
            f_ox = [i[0][0] for i in feature_ox.flat]
            f_oy = [i[0][0] for i in feature_oy.flat]
            f_oz = [i[0][0] for i in feature_oz.flat]

            feature = []

            if include_orientation:
                feature = feature+f_ow+f_ox+f_oy+f_oz
            if include_position:
                feature = feature+f_x+f_y+f_z
            if include_color:
                feature = feature+f_c_x+f_c_y
            if include_depth:
                feature = feature+f_d_x+f_d_y

            # feature = f_x+f_y+f_z+f_c_x+f_c_y+f_d_x+f_d_y+f_ow+f_ox+f_oy+f_oz

            features.append(feature)

        array_features = np.array(features)
        sequence_length = (array_features.shape[0])

        label_seq.append(np.array([label]*array_features.shape[0]))
        seq_len.append(sequence_length)
        all_activities_features.append(array_features)

        record_feature = {}
        record_feature['seq'] = _bytes_feature(array_features.tobytes())
        record_feature['label'] = _int64_feature(label)
        record_feature['seq_len'] = _int64_feature(sequence_length)

        example_features = tf.train.Features(feature=record_feature)
        example = tf.train.Example(features=example_features)
        writer.write(example.SerializeToString())

    writer.close()

def _parse_function(example_proto):
    features = {"seq": tf.FixedLenFeature([], tf.string, default_value=""),
              "label": tf.FixedLenFeature([], tf.int64, default_value=0),
              "seq_len": tf.FixedLenFeature([], tf.int64, default_value=0)}

    example = tf.parse_single_example(example_proto, features)
    array_features = tf.decode_raw(example['seq'],tf.float64)
    parsed_seq_len = example['seq_len']
    parsed_label = example['label']
    return array_features, parsed_label, parsed_seq_len
