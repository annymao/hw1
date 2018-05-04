'''Example code 4 :  classification using MFCC and DNN

Train your own deep learning model
for instrument classification.

Xuei-Wei Liao
'''
import model
import tensorflow as tf
import numpy as np
from time import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.metrics import classification_report

dataset_path = 'feature_label_data/dataset_mfcc_delta_13.npz'
dataset = np.load(dataset_path)
train = dataset['train']
train_label = dataset['ans_train'] 
test = dataset['test'].ravel()
test_label = dataset['ans_test'].ravel()
feature = np.vstack
# transform label to one-hot encoding
label = np.hstack((train_label,test_label))

trlb = preprocessing.LabelBinarizer()
trlb.fit(np.unique(label))
train_label_data = trlb.transform(train_label)
train_labels = train_label_data

telb = preprocessing.LabelBinarizer()
telb.fit(np.unique(label))
test_label_data = trlb.transform(test_label)
test_labels = test_label_data
"""
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
for tr_ind, ts_ind in sss.split(feature, label):
    train_examples = np.asarray(feature[tr_ind])
    train_labels = label_data[tr_ind]
    test_examples = feature[ts_ind]
    test_labels = label_data[ts_ind]
"""



# parameters for training
batch_size = 50
num_epochs = 500
num_class = np.unique(label).size
print(num_class)

# Parameters for Adam optimizer
init_learning_rate = 1e-4
epsilon = 1e-6
print(feature.shape)

def main(_):
    with tf.Graph().as_default(), tf.Session() as sess:
        # define your own fully connected DNN
        output = model.mydnn([256, 128, 128], num_class,
                             f_dim=feature.shape[1])

        # tensor for prediction the class
        prediction = tf.argmax(output, -1)
        # Add training ops into graph.
        with tf.variable_scope('train'):
            # tensor for labels
            label = tf.placeholder(
                tf.float32, shape=(None, num_class), name='labels')

            # tensor for calculate loss by softmax cross-entroppy
            
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=label,
                logits=output,
                name='loss_op'
            ))
            global_step = tf.Variable(
                0, name='global_step', trainable=False,
                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                             tf.GraphKeys.GLOBAL_STEP])
            optimizer = tf.train.AdamOptimizer(
                learning_rate=init_learning_rate,
                epsilon=epsilon)
            train_op = optimizer.minimize(
                loss, global_step=global_step, name='train_op')
            arg_label = tf.argmax(label, -1)

            acc = tf.reduce_mean(
                tf.cast(tf.equal(prediction, arg_label), tf.float32), name='acc_op')
            tf.summary.scalar('cross_entropy', tf.cast(loss, tf.float32))
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('train_log/',
                                                 sess.graph)
        sess.run(tf.global_variables_initializer())

        # Assign the required tensors to do the operation

        global_step_tensor = sess.graph.get_tensor_by_name(
            'train/global_step:0')
        features_tensor = sess.graph.get_tensor_by_name(
            'mydnn/input_features:0')
        train_op = sess.graph.get_operation_by_name('train/train_op')
        acc_op = sess.graph.get_tensor_by_name('train/acc_op:0')
        loss_tensor = sess.graph.get_tensor_by_name('train/loss_op:0')
        labels_tensor = sess.graph.get_tensor_by_name('train/labels:0')

        # Start training
        print('Start training...')
        print('Using dataset is: ' + dataset_path)
        t0 = time()
        epo = 0
        while epo < num_epochs:
            st = 0
            for _ in range(round(train_examples.shape[0] / batch_size)):
                [num_steps, _, loss_out] = sess.run([global_step_tensor, train_op, loss_tensor], feed_dict={features_tensor: train_examples[
                    st:st + batch_size], labels_tensor: train_labels[st:st + batch_size]})
                st += batch_size
            [summary] = sess.run([merged], feed_dict={
                                 features_tensor: train_examples, labels_tensor: train_labels})
            print('loss:', '%g' % np.mean(loss_out))
            [acc, p] = sess.run([acc_op, prediction], feed_dict={
                features_tensor: test_examples, labels_tensor: test_labels})
            train_writer.add_summary(summary, epo)
            epo += 1
            print("# of epochs: ", epo, ', test accuracy : ', '%.4f' % (acc))
        print('Finish training in {:4.2f} sec!'.format(time() - t0))
        print('Now test the trained DNN model....\n')
        print(classification_report(np.argmax(test_labels, -1), p))


if __name__ == '__main__':
    tf.app.run()
