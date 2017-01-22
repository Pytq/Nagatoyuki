import tensorflow as tf
from DeepPython import ToolBox, Model as M
from DeepPython import Elostd, Params


class Elodim(M.Model):
    def meta_params(self):
        return ['alpha']

    def linear_meta_params(self):
        return ['bais_ext']

    def features_data(self):
        return ['team_h', 'team_a', 'saison', 'journee', 'res']

    def get_prediction(self, s, target):
        n = 2
        self.m = [Elostd.Elostd(data_dict=self.data_dict) for _ in range(n)]
        for model in self.m:
            model.set_params(Params.paramStd)
        r = [x.get_prediction(s, target) for x in self.m]
        r = [tf.pow(x, self.metaparam['alpha']) for x in r]
        rf = tf.add_n(r)
        return rf/tf.reduce_sum(rf)

    def get_regularizer(self):
        list = [x.get_regularizer() for x in self.m]
        print(list)
        return tf.add_n(list)
