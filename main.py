import tensorflow as tf
import numpy as np
import util
import tf_modules as tfm
rng = np.random
tfd = tf.contrib.distributions

dat = util.get_mnist()
DX = dat.train.images.shape[1]
DY = dat.train.labels.shape[1]

###### BUILD MODELS
with tf.name_scope('ph') as scope:
    x_ph = tfm.ph((None, DX))  # bs x ntssi x wtss
    y_ph = tfm.ph((None, DY))  # bs x dy
    is_training_ph = tfm.ph(None, tf.bool)

# latent representation of the the shape of individual tss sites
with tf.variable_scope('encoder') as scope:
    h1 = tfm.fc_layer(x_ph,
                      dim_output=128,
                      act_fn=tf.nn.relu,
                      batch_norm=True,
                      p_drop=0.2,
                      is_training=is_training_ph, name='h1')
    # h2 = tfm.fc_layer(h1, 128, batch_norm=True, p_drop=0.2, is_training=is_training_ph, name='h2')
    y_hat_logit = tfm.linear(h1, DY, name='y_hat_logit')

losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat_logit, labels=y_ph)
loss = tf.reduce_mean(losses)

trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)



def prep_fd(d0, i_batch, is_training):
    x0 = d0.images[i_batch]
    y0 = d0.labels[i_batch]

    return {x_ph: x0,
            y_ph: y0,
            is_training_ph: is_training}

BATCH_SIZE = 16
batcher = util.Batcher(dat.train.num_examples, BATCH_SIZE)
N_STEP = int(1e4)
TEST_EVERY = int(1e3)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i_step in range(N_STEP):
        if i_step % TEST_EVERY == 0:
            test_batch = rng.choice(dat.test.num_examples, 100, replace=False)
            fd = prep_fd(dat.test, test_batch, False)
            yh0, l0 = sess.run([y_hat_logit, loss], feed_dict=fd)
            correct = yh0.argmax(1) == fd[y_ph].argmax(1)
            n_correct = correct.sum()
            acc0 = n_correct / yh0.shape[0]
            print('\ntest loss step {:d}: {:.2f}'.format(i_step, l0))
            print('test accuracy step {:d}: {:.2f}'.format(i_step, acc0))

            train_batch = rng.choice(dat.train.num_examples, 100, replace=False)
            fd = prep_fd(dat.train, train_batch, False)
            yh0, l0 = sess.run([y_hat_logit, loss], feed_dict=fd)
            correct = yh0.argmax(1) == fd[y_ph].argmax(1)
            n_correct = correct.sum()
            acc0 = n_correct / yh0.shape[0]
            print('train loss step {:d}: {:.2f}'.format(i_step, l0))
            print('train accuracy step {:d}: {:.2f}\n'.format(i_step, acc0))

        i_batch = batcher()
        fd = prep_fd(d0=dat.train,
                     i_batch=i_batch,
                     is_training=True)
        _, l0 = sess.run([trainer, loss], feed_dict=fd)
        print('loss step {:d}: {:.2f}'.format(i_step, l0), end='\r')


