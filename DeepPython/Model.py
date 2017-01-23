import tensorflow as tf
from DeepPython import ToolBox


class Model:
    def __init__(self, data_dict=None, targets=None, customizator=None, name='untitle', session=None):

        if session is None:
            raise Exception('A session must be fed')
        else:
            self.session = session
        if targets is None:
            targets = ['res']
        if customizator is None:
            customizator = {}
        self.customizator = customizator

        self.data_dict = data_dict
        self.name = name
        self.costs = {}
        self.train_step = {}
        self.dictparam = None

        # Define trainable_parameters
        self.trainable_params = {}
        self.define_parameters()

        # Define given_parameters
        self.given_params = {}
        self.ph_metaparam = {}
        for key in self.meta_params():
            self.ph_metaparam[key] = tf.placeholder(tf.float32)
            if key in self.linear_meta_params():
                self.given_params[key] = self.ph_metaparam[key]
            else:
                self.given_params[key] = tf.exp(self.ph_metaparam[key])

        # Define slice node
        self.current_slice = {}
        self.ph_current_slice = {}
        self.tf_assign_slice = []
        for key in self.features_data():
            self.ph_current_slice[key] = tf.placeholder(dtype=tf.float32)
            self.current_slice[key] = tf.Variable(self.ph_current_slice[key],
                                                  validate_shape=False, trainable=False, collections=[])
            assign_slice = tf.assign(self.current_slice[key], self.ph_current_slice[key], validate_shape=False)
            self.tf_assign_slice.append(assign_slice)

        # Define reset_trainable_parameters node
        params = [self.trainable_params[key] for key in self.trainable_params]
        self.reset_op = None
        if params != []:
            self.reset_op = tf.variables_initializer(params, name='{}.reset_trainable_parameters'.format(self.name))

        # Define prediction node
        self.prediction = {}
        for key in targets:
            self.prediction[key] = self.get_prediction(self.current_slice, key)

        # Define regularizer
        self.regulizer = self.get_regularizer()

    def finish_init(self):
        print('Model {} created. {} nodes in tf.Graph'.format(self.name, ToolBox.nb_tf_op()))
        if self.reset_op is not None:
            self.reset_trainable_parameters()

    def add_cost(self, cost, trainable=False):
        if cost.name in self.costs:
            print('Cost with name {} is already defined'.format(cost.name))
        else:
            self.costs[cost.name] = {}
            self.costs[cost.name]['cost'] = cost.get_cost(self)
            if trainable and not self.is_trainable():
                raise Exception('Model {} is not trainable yet is given cost {} which is trainable'.format(self.name, cost.name))
            elif trainable and self.is_trainable():
                self.costs[cost.name]['train_step'] = tf.train.AdamOptimizer(0.01).minimize(self.costs[cost.name])


    def set_params(self, param):
        self.dictparam = {self.ph_metaparam[key]: value for key, value in param.items() if key in self.ph_metaparam}

    def reset_trainable_parameters(self):
        if self.reset_op is not None:
            self.session.run(self.reset_op)
        else:
            raise Exception('Cannot reset trainable parameters of model {} which has no such parameters'.format(self.name))

    def train(self, cost, iterations):
        for _ in range(iterations):
            self.run(self.costs[cost]['train_step'])

    def get_cost(self, cost):
        return self.run(self.costs[cost]['cost'])

    def run(self, op_to_run):   # op is a tensorflow node and stands for operator.
        if self.dictparam is not None:
            return self.session.run(op_to_run, feed_dict=self.dictparam)
        else:
            raise Exception('Given parameters must be initialized for model: {}'.format(self.name))

    def meta_params(self):
        return []

    def features_data(self):
        raise NotImplementedError

    def linear_meta_params(self):
        return []

    def define_parameters(self):
        pass

    def get_prediction(self, s, target):
        raise NotImplementedError

    def get_regularizer(self):
        return tf.constant(0.)

    def is_trainable(self):
        return self.trainable_params != {}
