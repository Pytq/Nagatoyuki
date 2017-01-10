import tensorflow as tf
import ToolBox


class Model:
    def __init__(self, data_dict=None, targets=None, customizator=None):

        if targets is None:
            targets = ['res']
        if customizator is None:
            customizator = {}
        self.customizator = customizator

        self.data_dict = data_dict

        self.costs = {}
        self.train_step = {}
        self.session = None
        self.dictparam = {}
        self.init_all = None

        self.metaparam = {}
        self.ph_metaparam = {}
        for key in self.meta_params():
            if key in self.linear_meta_params():
                self.ph_metaparam[key] = tf.placeholder(tf.float32)
                self.metaparam[key] = self.ph_metaparam[key]
            else:
                self.ph_metaparam[key] = tf.placeholder(tf.float32)
                self.metaparam[key] = tf.exp(self.ph_metaparam[key])

        self.current_slice = {}
        self.ph_current_slice = {}
        self.tf_assign_slice = []
        for key in self.features_data():
            self.ph_current_slice[key] = tf.placeholder(dtype=tf.float32)
            self.current_slice[key] = tf.Variable(self.ph_current_slice[key],
                                                  validate_shape=False, trainable=False, collections=[])
            assign_slice = tf.assign(self.current_slice[key], self.ph_current_slice[key], validate_shape=False)
            self.tf_assign_slice.append(assign_slice)

        self.param = {}
        self.define_parameters()

        params = [self.param[key] for key in self.param]
        self.reset_op = tf.variables_initializer(params, name='init')

        self.prediction = {}
        for key in targets:
            self.prediction[key] = self.get_prediction(self.current_slice, key)

        self.regulizer = self.get_regularizer()

    def finish_init(self):
        # Create the session
        self.init_all = tf.global_variables_initializer()
        print('Model created. {} nodes in tf.Graph'.format(ToolBox.nb_tf_op()))
        self.new_session()

    def add_cost(self, cost, trainable=False):
        if cost.name in self.costs:
            print('cost already defined')
        else:
            self.costs[cost.name] = cost.get_cost(self)
            # print(self.costs)
            if trainable and self.is_trainable():
                self.train_step[cost.name] = tf.train.AdamOptimizer(0.01).minimize(self.costs[cost.name])

    def set_params(self, param):
        self.dictparam = {}
        for key in param:
            self.dictparam[self.ph_metaparam[key]] = param[key]

    def reset(self):
        self.session.run(self.reset_op)

    def new_session(self):
        if self.session is not None:
            self.session.close()
        self.session = tf.Session()
        self.session.run(self.init_all)

    def train(self, cost, iterations):
        for _ in range(iterations):
            self.run(self.train_step[cost])

    def get_cost(self, cost):
        # print(cost, self.costs[cost])
        return self.run(self.costs[cost])

    def run(self, x):
        return self.session.run(x, feed_dict=self.dictparam)

    def close(self):
        print('Session closed with {} nodes in tf.Graph'.format(ToolBox.nb_tf_op()))
        self.session.close()

    def meta_params(self):
        return []

    def features_data(self):
        raise NotImplementedError

    def linear_meta_params(self):
        raise NotImplementedError

    def define_parameters(self):
        return []

    def get_prediction(self, s, target):
        raise NotImplementedError

    def get_regularizer(self):
        return tf.constant(0.)

    def is_trainable(self):
        return True
