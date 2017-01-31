import tensorflow as tf
from DeepPython import Data, Params, ToolBox
import random
import numpy as np


class BasicRNNCell(tf.nn.rnn_cell.RNNCell):
    # state : nb_teams * 3 (elo win ; elo tie ; elo lose)
    # inputs * 2 : match (nb_teams * 2 (1,1)) ; resultat : nb_teams * 3 (1 sur l'evenement)
    # output : 3 (triplé ; pas forcément normalisé - [de proba])

    def __init__(self, nb_teams, keys):
        self.keys = keys
        self.__nb_teams = nb_teams
        self.STATE_SIZE = 1
        self.PREDICTION_SIZE = 3
        self.nb_debuggers = 5
        self.debugger = [tf.Variable(1.) for _ in range(self.nb_debuggers)]

    @property
    def state_size(self):
        return tf.TensorShape([self.__nb_teams, self.PREDICTION_SIZE])

    @property
    def output_size(self):
        return tf.TensorShape(1)

    def __call__(self, inputs, state, scope=None):
        # home_team: [batch, team]
        # away_team: [batch, team]
        # result: [batch, 3]
        # score:  [batch, (Params.MAX_GOALS+1)**2]
        # counted_for_loss: [batch, 1]

        # state: [batch, team, state_size]

        input_dict = self.get_input_dict(inputs)
        home_state = tf.batch_matmul(tf.expand_dims(input_dict['home_team'], axis=1), state)  #  [batch, 1, state_size]
        away_state = tf.batch_matmul(tf.expand_dims(input_dict['away_team'], axis=1), state)
        current_state = tf.concat(2, [home_state, away_state])


        u = tf.Variable(np.eye(3), trainable=False, dtype=tf.float32)
        u = tf.concat(0, [u,u])
        self.u = tf.Variable(np.eye(3), trainable=False, dtype=tf.float32)
        u = tf.expand_dims(tf.matmul(u, self.u), axis=0)
        current_state = tf.batch_matmul(current_state, u)
        # current_state = home_state + away_state
        current_state = tf.squeeze(current_state, axis=1)

        prediction = self.state_to_prediction(current_state)
        loss = self.debugger[3] * self.prediction_to_loss(input_dict, prediction)

        grad = self.debugger[4] * tf.gradients(tf.squeeze(loss), current_state)[0]

        output = tf.expand_dims(loss, axis=1)
        next_state = - self.debugger[2] * grad
        next_state = tf.expand_dims(next_state, axis=1)
        next_state = tf.batch_matmul(tf.expand_dims(input_dict['home_team'], axis=2), next_state)
        next_state = state + next_state

        return output, next_state

    def get_input_dict(self, inputs):
        input_dict = {self.keys[i]: inputs[i] for i in range(len(self.keys))}
        return input_dict

    def prediction_to_loss(self, input_dict, prediction):
        loss = self.debugger[0] * tf.reduce_sum(-tf.log(prediction + 1e-9) * input_dict['result'], axis=1)  # [batch]
        return loss

    def state_to_prediction(self, current_state):
        return self.debugger[1] * current_state


data = Data.Data(Params.FILE)
sess = tf.Session()

bashs = data.rnn_datas
types = data.rnn_datas_types
keys = list(bashs.keys())
inputs = [tf.Variable(initial_value=bashs[keys[i]], dtype=types[keys[i]], trainable=False) for i in range(len(keys))]

cell = BasicRNNCell(data.meta_datas["nb_teams"], keys)
loss, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, time_major=True)
loss = tf.reduce_sum(loss)


sess.run(tf.global_variables_initializer())

isOk = True
ll = sess.run(loss)
print('loss: ', ll)
var_list = cell.debugger + [cell.u]
for var in cell.debugger:
    sess.run(tf.assign(var, random.random()))
    ll2 = sess.run(loss)
    print('loss: ', ll2)
    isOk = isOk and ll2 != ll
    ll = ll2
grads = tf.gradients(loss, var_list)
print('grad: ', grads)
grads_values = sess.run(grads)
print('grad value: ', grads_values)
for val in grads_values:
    isOk = isOk and (val != 0.).any()
print()
if isOk:
    print("This is Ok :)")
else:
    print("WARNING NOT OK")
