import tensorflow as tf
import os, shutil

# from datatools import input_data
# mnist = input_data.read_data_sets("./data/", one_hot = True)
#
# learning_rate = 0.01
# batch_size = 100
# training_epoch = 60
# display_step = 1

n_hidden_1 = 256
n_hidden_2 = 256

def layer(x, w_shape, b_shape):
    W_init = tf.random_normal_initializer(stddev=(2.0/w_shape[0])**0.5)
    W = tf.get_variable('W', w_shape, initializer = W_init)
    b_init = tf.constant_initializer(value = 0)
    b = tf.get_variable('b', b_shape, initializer = b_init)
    output = tf.nn.relu(tf.matmul(x,W)+b)
    w_hist = tf.summary.histogram('weights',W)
    b_hist = tf.summary.histogram('biases', b)
    y_hist = tf.summary.histogram('output', output)
    return output

def inference(x):
    with tf.variable_scope('layer_1'):
        output_1 = layer(x, [784,n_hidden_1],[n_hidden_1])
    with tf.variable_scope('layer_2'):
        output_2 = layer(output_1, [n_hidden_1,n_hidden_2], [n_hidden_2])
    with tf.variable_scope('output_layer'):
        output = layer(output_2, [n_hidden_2,10], [10])
    return tf.nn.softmax(output)


# def loss(output, y):
#     xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y)
#     return tf.reduce_mean(xentropy)
#
# def training(cost, global_step):
#     tf.summary.scalar('cost', cost)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     return optimizer.minimize(cost, global_step = global_step)
#
# def evaluate(output, y):
#     correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar('validation error',(1.0-accuracy))
#     return accuracy

def check_digit_mlp(img_name):
    #if os.path.exists("logistic_logs/"):
        #shutil.rmtree("logistic_logs/")
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, 784])
        # y = tf.placeholder(tf.float32, shape=[None, 10])
        output = inference(x)
        # cost = loss(output, y)
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        # train_op = training(cost, global_step)
        # eval_op = evaluate(output, y)
        # summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        sess = tf.Session()
        # init = tf.global_variables_initializer()
        # if os.path.exists("mlp_logs/"):
        #     saver.restore(sess, tf.train.latest_checkpoint('./mlp_logs/'))

        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(os.path.dirname(base_dir), "multilayer/mlp_logs/")
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        result = sess.run(tf.argmax(output,1), feed_dict={x: conv_mnist(img_name)})
        print("Result:", result)
        return result
        #     print('model restored.')
        # else :
        #     sess.run(init)
        # summary_writer = tf.summary.FileWriter("logistic_logs/",
        #                                        graph = sess.graph)
        # for epoch in range (training_epoch):
        #     avg_cost = 0
        #     total_batch = int(mnist.train.num_examples/batch_size)
        #     for i in range(total_batch):
        #         minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
        #         sess.run(train_op, feed_dict = {x: minibatch_x,y: minibatch_y})
    #             avg_cost += sess.run(cost, feed_dict = {x: minibatch_x,y: minibatch_y})/total_batch
    #         if epoch%display_step == 0:
    #             print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))
    #             accuracy = sess.run(eval_op, feed_dict = {x: minibatch_x,y: minibatch_y})
    #             print("Validation Error:", (1-accuracy))
    #             summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
    #             summary_writer.add_summary(summary_str, sess.run(global_step))
    #             saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)
    # print('Optimization Finished.')
    # accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    # print("Test Accuracy:",accuracy)
