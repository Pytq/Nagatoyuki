import tensorflow as tf
from DeepPython import ToolBox, Model as M, Params, Bookmaker, Elostd, Elostdid


class Regresseur(M.Model):
    def features_data(self):
        return ['odd_win_h', 'odd_tie', 'odd_los_h', 'team_h', 'team_a', 'saison', 'journee', 'res']

    def define_parameters(self):
        self.param['alpha'] = tf.Variable(0.5)
        self.param['beta'] = tf.Variable(0.5)
        self.model__ = []
        for i in range(2):
            self.model__.append(Elostd.Elostd(data_dict=self.data_dict))
            self.model__[i].set_params(Params.paramStd)

        self.dictparam = {key:value for key,value in list(self.model__[0].dictparam.items()) + list(self.model__[1].dictparam.items())}
        self.model__b = Bookmaker.Bookmaker(data_dict=self.data_dict, customizator={'normalized': True}, name='book_reg')

    def get_prediction(self, s, target):
        if target == 'res':
            pred_model = self.param['beta'] * self.model__[0].get_prediction(s, target) + (1. - self.param['beta']) * self.model__[1].get_prediction(s, target)
            pred_book = self.model__b.get_prediction(s, target)
            pred = (self.param['alpha'] * pred_book + (1. - self.param['alpha']) * pred_model)
            return pred
        else:
            raise Exception(target + ' not implemented.')

    def get_regularizer(self):
        return self.model__[0].get_regularizer() + self.model__[1].get_regularizer()
