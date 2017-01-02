import tensorflow as tf


class Cost:
    def __init__(self, name='untitled'):
        self.name = name
        self.cost = None

    def define_logloss(self, target, feature_dim, regularized=False):
        if self.name is None:
            self.name = 'll_' + target
            if regularized:
                self.name = 'r' + self.name
        logloss = lambda model: self.cost_logloss(model.prediction[target], model.current_slice[target], feature_dim)
        if regularized:
            self.cost = lambda model: logloss(model) + model.regulizer
        else:
            self.cost = logloss

    def cost_logloss(self, prediction, target, feature_dim):
        reduction_indices = [i + 1 for i in range(feature_dim)]
        probabilities = prediction * target
        probabilities = tf.reduce_sum(probabilities, reduction_indices=reduction_indices)
        return tf.reduce_mean(-tf.log(probabilities + 1e-9))

    def apply(self, other, fun):
        result = Cost()
        if type(other) in [type(0), type(0.)]:
            result.name = self.name + '+' + str(other)
            result.cost = lambda m: fun(self.cost(m), other)
        else:
            result.name = self.name + '+' + other.name
            result.cost = lambda m: fun(self.cost(m), other.cost(m))
        return result

    def __add__(self, other):
        return self.apply(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        self.apply(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.__mul__(other)

    def get(self, s):
        return self.costs[s]



