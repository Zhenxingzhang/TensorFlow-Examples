import tensorflow as tf


if __name__ == "__main__":
    global_step = tf.Variable(0, trainable=False)
    inc = tf.assign_add(global_step, 1, name='increment')

    starter_learning_rate = 0.1

    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               3, 0.96, staircase=True)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(10):
            lr = sess.run([inc, learning_rate])
            print(lr)
