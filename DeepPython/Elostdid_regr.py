import tensorflow as tf
from DeepPython import ToolBox, Model as M


class Elostdid_regr(M.Model):

    def meta_params(self):
        if self.customizator['trainit']:
            return ['metaparam0', 'metaparamj0', 'metaparam2', 'metaparamj2']
        else:
            return ['metaparam0', 'metaparamj0', 'bais_ext', 'draw_elo', 'metaparam2', 'metaparamj2']

    def linear_meta_params(self):
        if self.customizator['trainit']:
            return []
        else:
            return ['bais_ext']

    def features_data(self):
        return ['team_h', 'team_a', 'saison', 'res', 'home_matchid', 'away_matchid', 'odds']

    def define_parameters(self):
        self.trainable_params['elo'] = tf.Variable(tf.zeros([self.data_dict["nb_teams"], self.data_dict["nb_saisons"]]))
        self.trainable_params['elojournee'] = tf.Variable(tf.zeros([self.data_dict["nb_teams"], self.data_dict["max_match_id"]]))
        self.trainable_params['bais_ext'] = tf.Variable(0.)
        self.trainable_params['draw_elo'] = tf.Variable(0.)
        self.trainable_params['alpha_e'] = tf.Variable(0.001, trainable=True)
        self.trainable_params['alpha_b'] = tf.Variable(1.0, trainable=True)

    def get_prediction(self, s, target):
        if target == 'res':
            elomatch = ToolBox.get_elomatch(s['team_h'] - s['team_a'], s['saison'], self.trainable_params['elo'])
            elomatch += ToolBox.get_elomatch(s['team_h'], s['home_matchid'], self.trainable_params['elojournee'])
            elomatch -= ToolBox.get_elomatch(s['team_a'], s['away_matchid'], self.trainable_params['elojournee'])
            if self.customizator['trainit']:
                elomatch += self.trainable_params['bais_ext']
                draw_elo = tf.exp(self.trainable_params['draw_elo'])
            else:
                elomatch += self.given_params['bais_ext']
                draw_elo = self.given_params['draw_elo']
            elomatch_win = elomatch - draw_elo
            elomatch_los = elomatch + draw_elo
            p_win = 1/(1. + tf.exp(-elomatch_win))
            p_los = 1. - 1/(1. + tf.exp(-elomatch_los))
            p_tie = 1. - p_los - p_win
            probas = tf.pack([p_win, p_tie, p_los], axis=1)

            merlin_res=(probas +s['odds'])/2
            ln_proba = tf.log(probas)
            ln_book = tf.log(s['odds'])
            #return tf.exp(self.trainable_params['alpha_e']*ln_proba + self.trainable_params['alpha_b']*ln_book)
            return merlin_res

        else:
            raise Exception(target + ' not implemented.')

    def get_regularizer(self):
        regulizer_list = []
        cost = ToolBox.get_raw_elo_cost(self.given_params['metaparam0'], 0,
                                        self.trainable_params['elo'], self.data_dict["nb_saisons"])
        regulizer_list.append(cost)

        cost = ToolBox.get_raw_elo_cost(self.given_params['metaparamj0'], 0,
                                        self.trainable_params['elojournee'], self.data_dict["max_match_id"])
        regulizer_list.append(cost)

        time_multiplicator = tf.constant(self.data_dict['time_diff'])
        cost = ToolBox.get_timediff_elo_cost(self.given_params['metaparam2'], self.trainable_params['elo'], self.data_dict["nb_saisons"])
        regulizer_list.append(cost)

        cost = ToolBox.get_timediff_elo_cost(self.given_params['metaparamj2'],
                                             self.trainable_params['elojournee'], self.data_dict["max_match_id"], tm = time_multiplicator)
        regulizer_list.append(cost)

        return tf.add_n(regulizer_list)

