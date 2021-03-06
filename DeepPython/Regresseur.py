import copy
import tensorflow as tf
from DeepPython import ToolBox, Model as M, Params, Bookmaker, Elostd, Elostdid


class Regresseur(M.Model):
    def features_data(self):
        features = []
        for m in self.model_list:
            features += m.features_data()
        return list(set(features))

    def define_parameters(self):
        self.model_list = []
        m = Elostdid.Elostd(data_dict=self.data_dict, name="elostd", session=self.session)
        m.set_params(Params.paramStd)
        self.model_list.append(m)
        self.model_list.append(Bookmaker.Bookmaker(data_dict=self.data_dict, customizator={'normalized': True}, name='book_reg', session=self.session))
        self.dictparam = {}
        for m in self.model_list:
            self.dictparam.update(m.dictparam)
            if 'alpha_' + m.name in self.trainable_params:
                raise Exception("Error model {} appears twice in regresseur".format(m.name))
            self.trainable_params['alpha_' + m.name] = tf.Variable(1. / len(self.model_list))

    def get_prediction(self, s, target):
        if target == 'res':
            sum_alpha = tf.add_n([self.trainable_params['alpha_' + m.name] for m in self.model_list])
            return tf.add_n([self.trainable_params['alpha_' + m.name] / sum_alpha * m.get_prediction(s, target) for m in self.model_list])
        else:
            raise Exception(target + ' not implemented.')

    def get_regularizer(self):
        return tf.add_n([m.get_regularizer() for m in self.model_list])
