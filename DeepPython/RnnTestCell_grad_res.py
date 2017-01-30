import tensorflow as tf
import random, copy
from DeepPython import Data, Params, ToolBox
import math
import numpy as np
import matplotlib.pyplot as plt

import time


PREDICTION_SIZE = 3


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
        self.STATE_SIZE = 1

        self.trainable_param = {}
        self.used_param = {}

        W = create_list([self.STATE_SIZE, PREDICTION_SIZE])

        b = create_list([1, PREDICTION_SIZE])
        b = [[ 0.35,-0.25,-0.35]]
        home_to_away = create_list([PREDICTION_SIZE, PREDICTION_SIZE])
        step_size = create_list([1, 2*self.STATE_SIZE], values=-3.)

        W[0][0] = 1.
        W[0][2] = -1.

        home_to_away[0][2] = 1.
        home_to_away[1][1] = 1.
        home_to_away[2][0] = 1.

        if (np.matmul(home_to_away, home_to_away) != np.identity(len(home_to_away))).all():
            raise Exception('Not involution transformation')

        stddev = 0.001

        home_to_away = tf.Variable(initial_value=home_to_away, trainable=False)

        self.trainable_param['W'] = tf.Variable(tf.truncated_normal([self.STATE_SIZE, PREDICTION_SIZE], stddev=stddev), trainable=False)
        W_tf = 0.01 * self.trainable_param['W'] + tf.Variable(initial_value=W, trainable=False)
        Wa_tf = tf.matmul(W_tf, home_to_away)
        self.used_param['W'] = tf.concat(0, [W_tf, Wa_tf])

        self.debugger = tf.Variable(1.)
        self.trainable_param['b'] =  tf.Variable(tf.truncated_normal([1, PREDICTION_SIZE], stddev=stddev), trainable=False)
        self.used_param['b'] = self.trainable_param['b'] + tf.Variable(initial_value=b, trainable=False)

        self.trainable_param['step_size'] = tf.Variable(tf.truncated_normal([1, 2*self.STATE_SIZE], stddev=stddev), trainable=False)
        step_size = tf.Variable(initial_value=step_size, trainable=False)
        self.used_param['step_size'] = tf.exp(self.trainable_param['step_size'] + step_size)

        # self.amixer = tf.Variable(0.1, trainable=False)
        # self.updateamixer = tf.assign(self.amixer, self.amixer / 1.01)

    @property
    def state_size(self):
        return tf.TensorShape([self.__nb_teams, self.STATE_SIZE])

    @property
    def output_size(self):
        return tf.TensorShape(1) #[self.__nb_teams, STATE_SIZE])

    def __call__(self, inputs, state, scope=None):
        # home_team: [batch, team]
        # away_team: [batch, team]
        # result: [batch, 3]
        # score:  [batch, (Params.MAX_GOALS+1)**2]
        # counted_for_loss: [batch, 1]

        # state: [batch, team, state_size]

        input_dict = self.get_input_dict(inputs)
        home_state = tf.batch_matmul(tf.expand_dims(input_dict['home_team'], axis=1), state)  #  [batch, 1, state_size]
        away_state = tf.batch_matmul(tf.expand_dims(input_dict['away_team'], axis=1), state)  # [batch, 1, state_size]
        current_state = tf.concat(2, [home_state, away_state])  # [batch, 1, 2*state_size]
        # current_state = tf.batch_matmul(home_state, tf.Variable([[[1., 0.]]], trainable=False))
        # current_state += tf.batch_matmul(away_state, tf.Variable([[[0., 1.]]], trainable=False))

        prediction = self.state_to_prediction(current_state)
        loss = self.debugger * self.prediction_to_loss(input_dict, prediction)

        grad = tf.gradients(tf.squeeze(loss), current_state)[0]
        print(grad)
        # grad = current_state * loss + 1
        step = - tf.map_fn(lambda batch: batch * self.used_param['step_size'], grad)


        # prediction2 = self.state_to_prediction(next_state1)

        home_next_state = tf.slice(step, [0, 0, 0], [-1, -1, self.STATE_SIZE])  # [batch, 1, state_size]
        away_next_state = tf.slice(step, [0, 0, self.STATE_SIZE], [-1, -1, self.STATE_SIZE])  # [batch, 1, state_size]
        # home_next_state = tf.batch_matmul(step, tf.Variable([[[1.], [0.]]], trainable=False))
        # away_next_state = tf.batch_matmul(step, tf.Variable([[[0.], [1.]]], trainable=False))

        home_next_state = tf.batch_matmul(tf.expand_dims(input_dict['home_team'], axis=2), home_next_state)
        away_next_state = tf.batch_matmul(tf.expand_dims(input_dict['away_team'], axis=2), away_next_state)

        next_state = state + home_next_state + away_next_state
        output = tf.expand_dims(loss, axis=1)

        return output, next_state

        # return output, tf.Print(next_state, [tf.squeeze(x) for x in
        #                              [state, input_dict['home_team'], input_dict['away_team'], current_state, next_state1, next_state]])

    def get_input_dict(self, inputs):
        input_dict = {self.keys[i]: inputs[i] for i in range(len(self.keys))}
        return input_dict

    def prediction_to_loss(self, input_dict, prediction):
        loss = tf.reduce_sum(-tf.log(prediction + 1e-9) * input_dict['result'], axis=1)  # [batch]
        return loss

    def state_to_prediction(self, current_state):
        transform = lambda batch: tf.matmul(batch, self.used_param['W']) + self.used_param['b']
        transformed_state1 = tf.map_fn(transform, current_state)  #  [batch, 1, prediction_size]
        transformed_state = tf.squeeze(transformed_state1, axis=1)  #  [batch, prediction_size]
        prediction1 = tf.exp(transformed_state)  # [batch, prediction_size]
        prediction = tf.map_fn(lambda batch: batch / tf.reduce_sum(batch), prediction1)  # [batch, prediction_size]
        return prediction


data = {}
data['train'] = Data.Data(Params.FILE)
#data['test'] = Data.Data(Params.FILE_TEST)
#data['test2'] = Data.Data(Params.FILE_TEST2)
sess = tf.Session()



loss = {}
cell = {}
for key in data:
    bashs = data[key].rnn_datas
    types = data[key].rnn_datas_types
    keys = list(bashs.keys())
    inputs = [tf.Variable(initial_value=bashs[keys[i]], dtype=types[keys[i]], trainable=False) for i in range(len(keys))]
    cell[key] = BasicRNNCell(data[key].meta_datas["nb_teams"], keys)
    loss[key], state = tf.nn.dynamic_rnn(cell[key], inputs, dtype=tf.float32, time_major=True)

# assign_op = []
# for batch_name in cell:
#     if batch_name != 'train':
#         for key in cell['train'].trainable_param:
#             assign_op.append(tf.assign(cell[batch_name].trainable_param[key], cell['train'].trainable_param[key]))

loss_train = tf.reduce_mean(loss['train']) # * bashs['counted_for_loss'])
# lossr = tf.reduce_mean(loss['train'])  # +  0.001 * (cell['train'].amixer ** 3) * cell['train'].reg
# loss_test = tf.reduce_mean(loss['test'][0])
#optimizer = tf.train.AdamOptimizer(0.1)
#opt_step = optimizer.minimize(loss_train, var_list=tf.trainable_variables())
# opt_matrix = tf.train.AdamOptimizer(0.1).minimize(cell['train'].reg)

print('START!')

def add_prints():
    for x in tf.get_default_graph().get_operations():
        inputs = x.inputs
        for y in inputs:
            y = tf.Print(y, [y])
            print(y)

sess.run(tf.global_variables_initializer())

print(sess.run(loss_train))
sess.run(tf.assign(cell['train'].debugger, 5.))
print(sess.run(loss_train))

loss_list = []

print(tf.trainable_variables())

grads_ = tf.gradients(loss_train, cell['train'].debugger)

print("yo", grads_)

exit()

for i in range(500):
    print()
    print('iteration number {}/250'.format(i))
    t = time.time()
    ll_train = sess.run(loss_train)
    ll_train  = 100 * ll_train * data['train'].meta_datas["nb_matchs"] / (data['train'].meta_datas["nb_matchs"] - 4950)
    print('train_nor', ll_train)
    loss_list.append(ll_train)
    for key, x in cell['train'].trainable_param.items():
        print(key, sess.run(x))
    print(tf.trainable_variables())
    # sess.run(opt_step)

    grads_and_vars = optimizer.compute_gradients(loss_train)
    grads = [g for g, _ in grads_and_vars]
    vars = [v for _, v in grads_and_vars]
    print(grads_and_vars)
    grad_vals = sess.run(grads)
    var_to_grad_val_map = {v.name: val for (v, val) in zip(vars, grad_vals)}
    print(var_to_grad_val_map)
    # sess.run(assign_op)
    print('time: {}'.format(time.time() - t))
    if False:
        plt.plot(loss_list)
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


