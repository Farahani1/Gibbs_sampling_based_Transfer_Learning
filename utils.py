def model_trainer(data, source_train, source_test, epochs=3000, verbose=True):

    import tensorflow as tf
    import numpy as np

    tf.reset_default_graph()
    inp = tf.placeholder(tf.float32, shape=[None, 784])
    x = tf.reshape(inp, shape=[-1, 28, 28, 1])
    lab = tf.placeholder(tf.float32, shape=[None, 10])
    l1 = tf.layers.conv2d(inputs=x, filters=6, kernel_size=5, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(l1, 2, 2)
    l2 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=5, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(l2, 2, 2)
    fc1 = tf.layers.flatten(conv2)
    fc2 = tf.layers.dense(fc1, 80, activation=tf.nn.relu)
    out = tf.layers.dense(fc2, 10, activation=tf.nn.sigmoid)
    batch_size = 512
    learning_rate = tf.placeholder(tf.float32, [])
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lab, logits=out))
    opt = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(error)
    init = tf.global_variables_initializer()

    # with tf.Session() as sess:
    sess = tf.Session()

    sess.run(init)
    for i in range(epochs):
        p = float(i) / epochs
        l = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.01 / (1. + 10 * p) ** 0.75

        lr = 0.0005
        ind = np.random.randint(0, source_train.shape[0], size=batch_size)
        x = source_train[ind, :, :].reshape([batch_size, 784])
        sess.run(opt, feed_dict={inp: x, lab: data['mnist_train_labels'][ind], learning_rate: lr})
        if (i % 100) == 0 and verbose:
            y_pred = sess.run(out, feed_dict={inp: source_test[0:5000, :, :].reshape([5000, 784])})
            cp = tf.equal(tf.argmax(out, 1), tf.argmax(lab, 1))

            acc = tf.reduce_mean(tf.cast(cp, tf.float32))
            print(sess.run(acc, feed_dict={out: y_pred, lab: data['mnist_test_labels'][0:5000]}))
        # print(sess.run(error,feed_dict={inp:x,lab:data.mnist_train_labels[ind]}))
    # del data
    print("Training is done!")
    return sess, out, inp


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          fig_size=(8, 6)):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=fig_size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def display_images(images, fig_size=(12, 12), m=8, n=8):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=fig_size)
    for i in range(m * n):
        sub = fig.add_subplot(m, n, i + 1)
        sub.axis('off')
        sub.imshow(images[i, :, :], interpolation='nearest', cmap='gray')


def report_accu(name, results, labels):
    from sklearn.metrics import accuracy_score

    pred = [res.argmax() for res in results]
    labels = [l.argmax() for l in labels[0:results.shape[0]]]

    print '\n The accuracy of ' + name + ' is :'
    print '     ' + str(accuracy_score(labels, pred))


def rgb2gray(rgb):

    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
