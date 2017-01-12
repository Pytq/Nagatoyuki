import tensorflow as tf


class Cost:

    def __init__(self, group, name=None, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}

        self.switcher = {
            'empty': self.__init_empty,
            'logloss': self.__init_logloss
        }

        if group not in self.switcher:
            raise Exception('Unkown type ' + group + 'in Cost.py')

        self.__parameters = feed_dict
        self.__group = group
        self.__cost = None
        self.name = name

        self.switcher[self.__group]()

    def get_cost(self, model):
        return self.__cost(model)

    def __add__(self, other):
        return self.__apply(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        self.__apply(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __init_empty(self):
        pass

    def __init_logloss(self):
        if self.name is None:
            self.name = 'll_' + self.__parameters['target']
            if self.__parameters['regularized']:
                self.name = 'r' + self.name
        self.__cost = self.__cost_logloss

    def __cost_logloss(self, m):
        reduction_indices = [i + 1 for i in range(self.__parameters['feature_dim'])]
        probabilities = m.prediction[self.__parameters['target']] * m.current_slice[self.__parameters['target']]
        probabilities = tf.reduce_sum(probabilities, reduction_indices=reduction_indices)
        if self.__parameters['regularized']:
            return tf.reduce_mean(-tf.log(probabilities + 1e-9)) + m.regulizer
        else:
            return tf.reduce_mean(-tf.log(probabilities + 1e-9))

    def __apply(self, other, fun):
        result = Cost()
        if type(other) in [type(0), type(0.)]:
            result.name = self.name + '+' + str(other)
            result.__cost = lambda m: fun(self.__cost(m), other)
        else:
            result.name = self.name + '+' + other.name
            result.__cost = lambda m: fun(self.__cost(m), other.cost(m))
        return result
