"""CapsNet model.

Related papers:
https://arxiv.org/pdf/1710.09829.pdf
http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf
"""
import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages


class CapsNet(object):
    """CapsNet model."""

    def __init__(self, hps, images, labels):
        """CapsNet constructor"""

        """
        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, channels]
          labels: Batches of labels. [batch_size, num_classes]
        """
        self.hps = hps
        self.images = images
        self.labels = labels

        self._extra_train_ops = []

    def _squash(self, x, axis=-1):
        """https://arxiv.org/pdf/1710.09829.pdf (eq.1)

           squash activation that normalizes vectors by their relative lengths
        """
        square_norm = tf.reduce_sum(tf.square(x), axis, keep_dims=True)
        scale = square_norm / (1 + square_norm) / tf.sqrt(square_norm + 1e-8)
        x = tf.multiply(scale, x)
        return x

    def _capsule_layer(self, x, params_shape, num_routing, name=''):
        """
        Credit XifengGuo: https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulelayers.py

        Xifeng Guo:
        The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
        neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
        from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
        [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

        :param num_routing: number of iterations for the routing algorithm
        """
        assert len(params_shape) == 4, "Given wrong parameter shape."
        output_num_capsule, input_num_capsule, output_dim_capsule, input_dim_capsule = params_shape

        x = self._batch_norm(name + 'bn', x)

        W = tf.get_variable(
            name + 'DW', params_shape, tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))

        inputs = x
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = tf.expand_dims(inputs, 1)
        tf.logging.info(f'Expanding inputs to be {inputs_expand.get_shape()}')

        # Replicate output_num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, output_num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = tf.tile(inputs_expand, [1, output_num_capsule, 1, 1])
        tf.logging.info(f'Tiling inputs to be {inputs_tiled.get_shape()}')

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[output_num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[output_num_capsule, input_num_capsule, output_dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [output_dim_capsule, input_dim_capsule]^T -> [output_dim_capsule].
        # inputs_hat.shape = [None, output_num_capsule, input_num_capsule, output_dim_capsule]
        inputs_hat = tf.map_fn(lambda x: tf.keras.backend.batch_dot(
            x, W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # In forward pass, `inputs_hat_stopped` = `inputs_hat`;
        # In backward, no gradient can flow from `inputs_hat_stopped` back to `inputs_hat`.
        inputs_hat_stopped = tf.stop_gradient(inputs_hat)

        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.output_num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[tf.shape(inputs_hat)[0],
                            output_num_capsule, input_num_capsule])

        assert num_routing > 0, 'The num_routing should be > 0.'
        for i in range(num_routing):
            # c.shape=[batch_size, output_num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
            if i == num_routing - 1:
                # c.shape =  [batch_size, output_num_capsule, input_num_capsule]
                # inputs_hat.shape=[None, output_num_capsule, input_num_capsule, output_dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [input_num_capsule] x [input_num_capsule, output_dim_capsule] -> [output_dim_capsule].
                # outputs.shape=[None, output_num_capsule, output_dim_capsule]
                outputs = self._squash(tf.keras.backend.batch_dot(
                    c, inputs_hat, [2, 2]))  # [None, 10, 16]
            else:  # Otherwise, use `inputs_hat_stopped` to update `b`. No gradients flow on this path.
                outputs = self._squash(tf.keras.backend.batch_dot(
                    c, inputs_hat_stopped, [2, 2]))

                # outputs.shape =  [None, output_num_capsule, output_dim_capsule]
                # inputs_hat.shape=[None, output_num_capsule, input_num_capsule, output_dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [output_dim_capsule] x [input_num_capsule, output_dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, output_num_capsule, input_num_capsule]
                b += tf.keras.backend.batch_dot(outputs, inputs_hat_stopped, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#
        tf.logging.info(f'image after this capsule layer {outputs.get_shape()}')
        return outputs

    def build_graph(self):
        """Build a whole graph for the model."""

        """
            1. create anxilliary parameters
            2. create CapsNet core layers
            3. create trainer
        """

        self._build_init()
        self._build_model()

        grads_vars = self.optimizer.compute_gradients(self.cost)

        self._build_train_op(grads_vars)
        self.summaries = tf.summary.merge_all()

    def _build_init(self):
        with tf.device('/cpu:0'):

            self.is_training = tf.placeholder(tf.bool)

            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.increase_global_step = self.global_step.assign_add(1)

            self.lrn_rate = tf.maximum(tf.train.exponential_decay(
                self.hps.lrn_rate, self.global_step, 1e3, 0.66), self.hps.min_lrn_rate)
            # self.lrn_rate = tf.Variable(self.hps.lrn_rate, dtype=tf.float32, trainable=False)
            tf.summary.scalar('learning_rate', self.lrn_rate)

            if self.hps.optimizer == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
            elif self.hps.optimizer == 'mom':
                self.optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
            elif self.hps.optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.lrn_rate)

    def _build_model(self):
        """Build the core model within the graph."""

        filters = self.hps.filters
        assert isinstance(filters, list)
        strides = self.hps.strides
        assert isinstance(strides, list)
        cnn_kernel_size = self.hps.cnn_kernel_size
        assert isinstance(cnn_kernel_size, int)
        padding = "VALID" if self.hps.standard else "SAME"

        with tf.variable_scope('init'):
            x = self.images
            x = self._batch_norm('init_bn', x)
            x = self._conv('init_conv', x, cnn_kernel_size,
                           filters[0], filters[1], self._stride_arr(strides[0]), padding=padding)
            x = self._relu(x)
            tf.logging.info(f'image after init {x.get_shape()}')

        with tf.variable_scope('primal_capsules'):
            x = self._batch_norm('primal_capsules_bn', x)
            x = self._conv('primal_capsules_conv', x, cnn_kernel_size, filters[1],
                           filters[1], self._stride_arr(strides[1]), padding=padding)
            x = self._squash(x)
            # 256 / 32 = 8
            capsules_dims = (filters[1] // filters[2])
            # 6x6x256 / 8 = 1152
            num_capsules = np.prod(x.get_shape().as_list()[1:]) // (filters[1] // filters[2])
            # TensorFlow does the trick
            x = tf.reshape(x, [-1, num_capsules, capsules_dims])
            tf.logging.info(f'image after primal capsules {x.get_shape()}')

        '''
        # EXPERIMENT: adding multilayer capsules
        with tf.variable_scope('digital_capsules_0'):
            params_shape = [128, num_capsules, filters[3], capsules_dims, capsules_dims]
            x = self._capsule_layer(x, params_shape=params_shape, num_routing=self.hps.num_routing)
        '''

        with tf.variable_scope('digital_capsules_final'):
            """
                params_shape = [output_num_capsule, input_num_capsule,
                                output_dim_capsule, input_dim_capsule]
            """
            params_shape = [self.hps.num_classes, num_capsules, filters[3], capsules_dims]
            x = self._capsule_layer(x, params_shape=params_shape, num_routing=self.hps.num_routing)

        # Compute length of each [None,output_num_capsule,output_dim_capsule]
        y_pred = tf.sqrt(tf.reduce_sum(tf.square(x), 2))

        with tf.variable_scope('costs'):
            """Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it."""
            y_true = self.labels
            L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
                0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

            ce = tf.reduce_mean(tf.reduce_sum(L, 1), name='cross_entropy')

            self.cost = ce + self._decay()
            tf.summary.scalar(f'Total_loss', self.cost)
            tf.summary.scalar(f'Cross_Entropy_loss', ce)

        with tf.variable_scope('acc'):
            correct_prediction = tf.equal(
                tf.argmax(y_pred, 1), tf.argmax(self.labels, 1))
            self.acc = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name='accu')
            tf.summary.scalar(f'accuracy', self.acc)

    def _build_train_op(self, grads_vars):
        """Build training specific ops for the graph."""
        '''
        # Add histograms for trainable variables.
        # Add histograms for gradients.
        for grad, var in grads_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name, var)
                tf.summary.histogram(var.op.name + '/gradients', grad)
        '''
        # defensive step 2 to clip norm
        clipped_grads, self.norm = tf.clip_by_global_norm(
            [g for g, _ in grads_vars], self.hps.global_norm)

        # defensive step 3 check NaN
        # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
        grad_check = [tf.check_numerics(g, message='NaN Found!') for g in clipped_grads]
        with tf.control_dependencies(grad_check):
            apply_op = self.optimizer.apply_gradients(
                zip(clipped_grads, [v for _, v in grads_vars]),
                global_step=self.global_step, name='train_step')

            train_ops = [apply_op] + self._extra_train_ops
            # Group all updates to into a single train op.
            self.train_op = tf.group(*train_ops)

    """Anxilliary methods"""

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_mean, mean, 0.99))
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_variance, variance, 0.99))
            tf.summary.histogram(moving_mean.op.name, moving_mean)
            tf.summary.histogram(moving_variance.op.name, moving_variance)

            def train():
                # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
                return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)

            def test():
                return tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, 0.001)
            y = tf.cond(tf.equal(self.is_training, tf.constant(True)), train, test)
            y.set_shape(x.get_shape())
            return y

    # override _conv to use He initialization with truncated normal to prevent dead neural
    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, padding='VALID'):
        """Convolution."""
        with tf.variable_scope(name):
            # n = filter_size * filter_size * out_filters
            n = in_filters + out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.truncated_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding=padding)

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _relu(self, x, leakiness=0.0, elu=False):
        """Relu, with optional leaky support."""
        if leakiness > 0.0:
            return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
        elif elu:
            return tf.nn.elu(x)
        else:
            return tf.nn.relu(x)

    def _fully_connected(self, x, out_dim, name=''):
        """FullyConnected layer for final output."""
        x = tf.contrib.layers.flatten(x)
        w = tf.get_variable(
            name + 'DW', [x.get_shape()[1], out_dim],
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True))
        b = tf.get_variable(name + 'biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def total_parameters(self, var_list=None):
        if var_list is None:
            return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() + [var for var in tf.global_variables() if 'bn' in var.name]]).astype(np.int32)
        else:
            return np.sum([np.prod(v.get_shape().as_list()) for v in var_list]).astype(np.int32)
