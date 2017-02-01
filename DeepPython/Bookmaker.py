from DeepPython import Model as M
import tensorflow as tf


class Bookmaker(M.Model):
    def features_data(self):
        return ['odd_win_h', 'odd_tie', 'odd_los_h', 'res']

    def is_trainable(self):
        return False

    def get_prediction(self, s, target):
        if target == 'res':
            p_win = 1. / tf.squeeze(s['odd_win_h'])
            p_tie = 1. / tf.squeeze(s['odd_tie'])
            p_los = 1. / tf.squeeze(s['odd_los_h'])
            if 'toNormalize' in self.customizator and not self.customizator['toNormalize']:
                return tf.pack([p_win, p_tie, p_los], axis=1)
            else:
                sum_p = p_win + p_los + p_tie
                return tf.pack([p_win / sum_p, p_tie / sum_p, p_los / sum_p], axis=1)

        else:
            print(target + ' not implemented.')