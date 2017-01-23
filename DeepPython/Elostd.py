import tensorflow as tf
from DeepPython import ToolBox, Model as M


class Elostd(M.Model):
    def __init__(self):
        raise Exception("Depreciated")

    def meta_params(self):
        return ['metaparam0', 'metaparam1', 'metaparamj0', 'metaparamj1',
                'metaparam2', 'metaparamj2', 'bais_ext', 'draw_elo']

    def linear_meta_params(self):
        return ['bais_ext']

    def features_data(self):
        return ['team_h', 'team_a', 'saison', 'journee', 'res']

    def define_parameters(self):
        self.trainable_params['elo'] = tf.Variable(tf.random_normal([self.data_dict["nb_teams"], self.data_dict["nb_saisons"]], mean=0.0, stddev=0.1)) #tf.zeros
        self.trainable_params['elojournee'] = tf.Variable(tf.random_normal([self.data_dict["nb_teams"], self.data_dict["nb_journee"]], mean=0.0, stddev=0.1))

    def get_prediction(self, s, target):
        if target == 'res':
            elomatch = ToolBox.get_elomatch(s['team_h'] - s['team_a'], s['saison'], self.trainable_params['elo'])
            elomatch += ToolBox.get_elomatch(s['team_h'] - s['team_a'], s['journee'], self.trainable_params['elojournee'])
            elomatch += self.given_params['bais_ext']
            elomatch_win = elomatch - self.given_params['draw_elo']
            elomatch_los = elomatch + self.given_params['draw_elo']
            p_win = 1/(1. + tf.exp(-elomatch_win))
            p_los = 1. - 1/(1. + tf.exp(-elomatch_los))
            p_tie = 1. - p_los - p_win
            return tf.pack([p_win, p_tie, p_los], axis=1)
        else:
            raise Exception(target + ' not implemented.')

    def get_regularizer(self):
        regulizer_list = []
        cost = ToolBox.get_raw_elo_cost(self.given_params['metaparam0'], self.given_params['metaparam1'],
                                        self.trainable_params['elo'], self.data_dict["nb_saisons"])
        regulizer_list.append(cost)

        cost = ToolBox.get_raw_elo_cost(self.given_params['metaparamj0'], self.given_params['metaparamj0'],
                                        self.trainable_params['elojournee'], self.data_dict["nb_journee"])
        regulizer_list.append(cost)

        cost = ToolBox.get_timediff_elo_cost(self.given_params['metaparam2'], self.trainable_params['elo'], self.data_dict["nb_saisons"])
        regulizer_list.append(cost)

        cost = ToolBox.get_timediff_elo_cost(self.given_params['metaparamj2'],
                                             self.trainable_params['elojournee'], self.data_dict["nb_journee"])
        regulizer_list.append(cost)

        return tf.add_n(regulizer_list)

