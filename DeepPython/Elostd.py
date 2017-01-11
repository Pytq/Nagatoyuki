import tensorflow as tf
from DeepPython import ToolBox, Model as M


class Elostd(M.Model):
    def meta_params(self):
        return ['metaparam0', 'metaparam1', 'metaparamj0', 'metaparamj1',
                'metaparam2', 'metaparamj2', 'bais_ext', 'draw_elo']

    def linear_meta_params(self):
        return ['bais_ext']

    def features_data(self):
        return ['team_h', 'team_a', 'saison', 'journee', 'res']

    def define_parmeters(self):
        self.param['elo'] = tf.Variable(tf.zeros([self.data_dict["nb_teams"], self.data_dict["nb_saisons"]]))
        self.param['elojournee'] = tf.Variable(tf.zeros([self.data_dict["nb_teams"], self.data_dict["nb_journee"]]))

    def get_prediction(self, s, target):
        if target == 'res':
            elomatch = ToolBox.get_elomatch(s['team_h'] - s['team_a'], s['saison'], self.param['elo'])
            elomatch += ToolBox.get_elomatch(s['team_h'] - s['team_a'], s['journee'], self.param['elojournee'])
            elomatch += self.metaparam['bais_ext']
            elomatch_win = elomatch - self.metaparam['draw_elo']
            elomatch_los = elomatch + self.metaparam['draw_elo']
            p_win = 1/(1. + tf.exp(-elomatch_win))
            p_los = 1. - 1/(1. + tf.exp(-elomatch_los))
            p_tie = 1. - p_los - p_win
            return tf.pack([p_win, p_tie, p_los], axis=1)
        else:
            raise Exception(target + ' not implemented.')

    def get_regularizer(self):
        regulizer_list = []
        cost = ToolBox.get_raw_elo_cost(self.metaparam['metaparam0'], self.metaparam['metaparam1'],
                                        self.param['elo'], self.data_dict["nb_saisons"])
        regulizer_list.append(cost)

        cost = ToolBox.get_raw_elo_cost(self.metaparam['metaparamj0'], self.metaparam['metaparamj0'],
                                        self.param['elojournee'], self.data_dict["nb_journee"])
        regulizer_list.append(cost)

        cost = ToolBox.get_timediff_elo_cost(self.metaparam['metaparam2'], self.param['elo'], self.data_dict["nb_saisons"])
        regulizer_list.append(cost)

        cost = ToolBox.get_timediff_elo_cost(self.metaparam['metaparamj2'],
                                             self.param['elojournee'], self.data_dict["nb_journee"])
        regulizer_list.append(cost)

        return tf.add_n(regulizer_list)

