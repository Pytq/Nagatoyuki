import tensorflow as tf
import copy
import ToolBox
import Costs
from abc import ABCMeta, abstractmethod


class Model(object,metaclass=ABCMeta):
    def __init__(self, data_dict = None):
        
        self.data_dict = data_dict
        
        self.costs = {}
        self.regulizer = {}
        self.train_step = {}
        self.session = None
        self.prediction = {}
        self.metaparam = {}
        self.dictparam = {}
        self.param = {}
        self.ph_metaparam = {}
        self.current_slice = {}
        self.ph_current_slice = {}
        self.tf_assign_slice = []

        # WTF les noms self.ph_current_slice et self.current_slice ?! 
        
        self.init_all = None

        for key in self.meta_params():
            if key in self.linear_meta_params():
                self.ph_metaparam[key] = tf.placeholder(tf.float32)
                self.metaparam[key] = self.ph_metaparam[key]
            else:
                self.ph_metaparam[key] = tf.placeholder(tf.float32)
                self.metaparam[key] = tf.exp(self.ph_metaparam[key])

        for key in self.features_data():
            self.ph_current_slice[key] = tf.placeholder(dtype=tf.float32)
            self.current_slice[key] = tf.Variable(self.ph_current_slice[key], validate_shape=False, 
                                                        trainable=False, collections=[])
            self.tf_assign_slice.append(tf.assign(self.current_slice[key], self.ph_current_slice[key],
                                                        validate_shape=False))
            
        self.define_parmeters()
        params = [self.param[key] for key in self.param]
        self.reset_op = tf.initialize_variables(params, name='init')
        self.prediction['res'] = self.get_prediction(self.current_slice, 'res')
        self.regulizer = self.get_regularizer()



    def finish_init(self):
        # Create the session
        self.init_all = tf.initialize_all_variables()
        print('Model created. ' + str(ToolBox.nb_tf_op()) + ' nodes in tf.Graph')
        self.new_session()

    def define_logloss(self, target='res', target_dim=1, name=None, regularized=False, trainable=False):
        cost = Costs.Cost(name=name)
        cost.define_logloss(target, target_dim, regularized=regularized)
        self.add_cost(cost, trainable=trainable)

    def add_cost(self, cost, trainable=False):
        if cost.name in self.costs:
            print('cost already defined')
        else:
            self.costs[cost.name] = cost.cost(self)
            if trainable:
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
        return self.run(self.costs[cost])

#    def shuffle(self):
#        self.session.run(self.data.shuffle) # NON NON

    def run(self, x):
        return self.session.run(x, feed_dict=self.dictparam)

    def close(self):
        print('Session closed with ' + str(ToolBox.nb_tf_op()) + ' nodes in tf.Graph')
        self.session.close()

    @abstractmethod
    def meta_params(self):
        pass

    @abstractmethod
    def features_data(self):
        pass

    @abstractmethod
    def linear_meta_params(self):
        pass

    @abstractmethod
    def define_parmeters(self):
        pass

    @abstractmethod
    def get_prediction(self, s):
        pass

    @abstractmethod
    def get_regularizer(self):
        pass
