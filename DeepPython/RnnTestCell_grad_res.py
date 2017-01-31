import tensorflow as tf
import random, copy
from DeepPython import Data, Params, ToolBox
import math
import numpy as np
import matplotlib.pyplot as plt

import time


PREDICTION_SIZE = (Params.MAX_GOALS+1) ** 2


def create_list(dims, random=False, values=0.):
    if dims == []:
        if random:
            return random.randint(1, 9)
        else:
            return values
    mydims = copy.deepcopy(dims)
    dim = mydims.pop(0)
    return [create_list(mydims, random=random, values=values) for _ in range(dim)]

def sign(x):
    if x > 0:
        return 1.
    elif x==0:
        return 0.
    else:
        return -1.

class BasicRNNCell(tf.nn.rnn_cell.RNNCell):
    # state : nb_teams * 3 (elo win ; elo tie ; elo lose)
    # inputs * 2 : match (nb_teams * 2 (1,1)) ; resultat : nb_teams * 3 (1 sur l'evenement)
    # output : 3 (triplé ; pas forcément normalisé - [de proba])

    def __init__(self, nb_teams, keys):
        self.keys = keys
        self.__nb_teams = nb_teams
        self.STATE_SIZE = 2

        self.trainable_param = {}
        self.used_param = {}
        self.fix_param = {}

        W = create_list([self.STATE_SIZE, PREDICTION_SIZE])
        b = create_list([PREDICTION_SIZE])
        home_to_away = create_list([PREDICTION_SIZE, PREDICTION_SIZE])
        step_size = create_list([self.STATE_SIZE], values=-10.)
        # step_size = [-8., -8.]
        score_to_res = create_list([PREDICTION_SIZE, 3])

        for goals_home in range(Params.MAX_GOALS+1):
            b[goals_home * (Params.MAX_GOALS + 1) + goals_home] += 0.
            for goals_away in range(Params.MAX_GOALS + 1):
                W[0][goals_away * (Params.MAX_GOALS+1) + goals_home] = float(goals_home)
                W[1][goals_away * (Params.MAX_GOALS+1) + goals_home] = -float(goals_away)
                # W[2][goals_away * (Params.MAX_GOALS+1) + goals_home] = sign(goals_home - goals_away)

                b[goals_away * (Params.MAX_GOALS + 1) + goals_home] += -math.log(float(math.factorial(goals_home) * math.factorial(goals_away)))

                home_to_away[goals_away * (Params.MAX_GOALS+1) + goals_home][goals_home * (Params.MAX_GOALS+1) + goals_away] = 1.

                score_to_res[goals_away * (Params.MAX_GOALS + 1) + goals_home][0] = 1. if goals_home > goals_away else 0.
                score_to_res[goals_away * (Params.MAX_GOALS + 1) + goals_home][1] = 1. if goals_home == goals_away else 0.
                score_to_res[goals_away * (Params.MAX_GOALS + 1) + goals_home][2] = 1. if goals_home < goals_away else 0.

        if (np.matmul(home_to_away, home_to_away) != np.identity(len(home_to_away))).all():
            raise Exception('Not involution transformation')

        stddev = 0.001

        self.fix_param['home_to_away'] = tf.Variable(initial_value=home_to_away, trainable=False)

        self.trainable_param['W'] = tf.Variable(tf.truncated_normal([self.STATE_SIZE, PREDICTION_SIZE], stddev=stddev), trainable=True)
        self.used_param['W'] = 0.01 * self.trainable_param['W'] + tf.Variable(initial_value=W, trainable=False)

        self.trainable_param['b'] =  tf.Variable(tf.truncated_normal([PREDICTION_SIZE], stddev=stddev), trainable=True)
        self.used_param['b'] = self.trainable_param['b'] + tf.Variable(initial_value=b, trainable=False)

        self.trainable_param['step_size'] = tf.Variable(tf.truncated_normal([self.STATE_SIZE], stddev=stddev), trainable=True)
        step_size = tf.Variable(initial_value=step_size, trainable=False)
        self.used_param['step_size'] = tf.exp(self.trainable_param['step_size'] + step_size)

        self.fix_param['score_to_res'] = tf.Variable(initial_value=score_to_res, trainable=False)

        self.trainable_param['mixer'] = tf.Variable(initial_value=0., trainable=False)
        self.used_param['mixer'] = 0.05 * self.trainable_param['mixer'] + tf.Variable(initial_value=0.8, trainable=False)

        self.trainable_param['goals'] = tf.Variable(0., trainable=False)
        self.used_param['goals'] = (0.71 + 0.01 * self.trainable_param['goals']) * tf.Variable([1., 0.], trainable=False)

        # self.amixer = tf.Variable(0.1, trainable=False)
        # self.updateamixer = tf.assign(self.amixer, self.amixer / 1.01)

    @property
    def state_size(self):
        return tf.TensorShape([self.__nb_teams, self.STATE_SIZE])

    @property
    def output_size(self):
        #return (tf.TensorShape(1), tf.TensorShape(3), tf.TensorShape(3))
        return (tf.TensorShape(1), tf.TensorShape(1)) #[self.__nb_teams, STATE_SIZE])

    def __call__(self, inputs, state, scope=None):
        input_dict = self.get_input_dict(inputs)
        home_state = tf.squeeze(tf.batch_matmul(tf.expand_dims(input_dict['home_team'], axis=1), state), axis=1)  # [batch, state_size]       batch team x batch team statesize
        away_state = tf.squeeze(tf.batch_matmul(tf.expand_dims(input_dict['away_team'], axis=1), state), axis=1)  # [batch, state_size]       batch state x batch team 1 1 state

        prediction = self.get_prediction(home_state, away_state)

        # prediction_final = tf.nn.softmax(prediction)
        loss_score = tf.nn.softmax_cross_entropy_with_logits(prediction, input_dict['score'])
        prediction_result = tf.matmul(prediction, self.fix_param['score_to_res'])
        loss_result = tf.nn.softmax_cross_entropy_with_logits(prediction_result, input_dict['result'])

        grads = tf.gradients(tf.squeeze(loss_score, axis=0), [home_state, away_state])

        steps = [- grad * self.used_param['step_size'] for grad in grads]


        home_step = tf.batch_matmul(tf.expand_dims(input_dict['home_team'], axis=2), tf.expand_dims(steps[0], axis=1))
        away_step = tf.batch_matmul(tf.expand_dims(input_dict['away_team'], axis=2), tf.expand_dims(steps[1], axis=1))
        next_state = state + home_step + away_step

        #return (tf.expand_dims(loss_result, axis=1), prediction_result, input_dict['result']), next_state
        #return prediction_result, next_state
        return (tf.expand_dims(loss_result, axis=1), tf.expand_dims(loss_score, axis=1)), next_state

    def get_input_dict(self, inputs):
        input_dict = {self.keys[i]: inputs[i] for i in range(len(self.keys))}
        return input_dict

    def get_prediction(self, home_state, away_state):
        home_state = tf.nn.bias_add(home_state, self.used_param['goals'])
        away_state = tf.nn.bias_add(away_state, self.used_param['goals'])
        home_prediction = tf.matmul(home_state, self.used_param['W'])
        away_prediction = tf.matmul(away_state, self.used_param['W'])
        away_prediction = tf.matmul(away_prediction, self.fix_param['home_to_away'])
        prediction = home_prediction + away_prediction
        prediction = tf.nn.bias_add(prediction, self.used_param['b'])
        return prediction


data = {}
data['train'] = Data.Data(Params.FILE)
data['test'] = Data.Data(Params.FILE_TEST)
#data['test2'] = Data.Data(Params.FILE_TEST2)
sess = tf.Session()



loss = {}
cell = {}
bashs = {}
loss_score = {}
loss_result = {}
for key in data:
    bashs[key] = data[key].rnn_datas
    types = data[key].rnn_datas_types
    keys = list(bashs[key].keys())
    inputs = [tf.Variable(initial_value=bashs[key][keys[i]], dtype=types[keys[i]], trainable=False) for i in range(len(keys))]
    cell[key] = BasicRNNCell(data[key].meta_datas["nb_teams"], keys)
    loss[key], state = tf.nn.dynamic_rnn(cell[key], inputs, dtype=tf.float32, time_major=True)
    # a,b,c = loss[key]
    loss_result[key], loss_score[key] = loss[key]

assign_op = []
for batch_name in cell:
    if batch_name != 'train':
        for key in cell['train'].trainable_param:
            assign_op.append(tf.assign(cell[batch_name].trainable_param[key], cell['train'].trainable_param[key]))

# sess.run(tf.global_variables_initializer())
#
# for x in [a,b,c]:
#     print(sess.run(x))
# exit()
loss_train = tf.reduce_sum(loss_result['train'] * bashs['train']['counted_for_loss'])/tf.reduce_sum(bashs['train']['counted_for_loss'])
# lossr = tf.reduce_mean(loss['train'])  # +  0.001 * (cell['train'].amixer ** 3) * cell['train'].reg
loss_test = tf.reduce_sum(loss_result['test'] * bashs['test']['counted_for_loss'])/tf.reduce_sum(bashs['test']['counted_for_loss'])
optimizer = tf.train.AdamOptimizer(0.1)
opt_step = optimizer.minimize(loss_train, var_list=tf.trainable_variables())
# opt_matrix = tf.train.AdamOptimizer(0.1).minimize(cell['train'].reg)

print('START!')

def add_prints():
    for x in tf.get_default_graph().get_operations():
        inputs = x.inputs
        for y in inputs:
            y = tf.Print(y, [y])
            print(y)

sess.run(tf.global_variables_initializer())

loss_list = {'train': [], 'test': []}
for i in range(500):
    print()
    print('iteration number {}/250'.format(i))
    t = time.time()
    ll_train = sess.run(loss_train)
    ll_train  *= 100
    ll_test = sess.run(loss_test)
    ll_test *= 100
    print('train', ll_train)
    print('test', ll_test)
    loss_list['train'].append(ll_train)
    loss_list['test'].append(ll_test)

    for key, x in cell['train'].used_param.items():
        print(key, sess.run(x))
    print(tf.trainable_variables())
    sess.run(opt_step)
    sess.run(assign_op)
    # grads_and_vars = optimizer.compute_gradients(loss_train)
    # grads = [g for g, _ in grads_and_vars]
    # vars = [v for _, v in grads_and_vars]
    # print(grads_and_vars)
    # grad_vals = sess.run(grads)
    # var_to_grad_val_map = {v.name: val for (v, val) in zip(vars, grad_vals)}
    # print(var_to_grad_val_map)

    print('time: {}'.format(time.time() - t))
    if True:
        plt.plot(loss_list['test'])
        plt.plot(loss_list['train'])
        plt.pause(0.001)
        plt.ion()
        plt.show()

print(loss_list)

plt.plot(loss_list)
plt.show()

sess.close()
exit()















teams = []
for i in range(len(final_state)):
    team = [data.id_to_team[i], int(400 * final_state[i][0] - final_state[i][2])]
    teams.append(team)

teams.sort(key=lambda x:-x[1])
for x in teams:
    print("{}: {}".format(x[0],x[1]))

exit()

for _ in range(5):
    print(sess.run(loss_total))
    for key, value in cell.trainable_param.items():
        print(key, sess.run(value))
    sess.run(optimizer)


print(sess.run(loss_total))

final_state = sess.run(final_state)

import matplotlib.pyplot as plt

x = []
y = []
for s in final_state:
    x.append(s[2] - s[0])
    y.append(s[1] - s[0])
plt.scatter(x,y)
plt.show()


