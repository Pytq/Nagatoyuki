import tensorflow as tf
import random, copy
from DeepPython import Data, Params

class BasicRNNCell(tf.nn.rnn_cell.RNNCell):
    # memoryState : nb_teams * 3 (elo win ; elo tie ; elo lose)
    # inputs * 2 : match (nb_teams * 2 (1,1)) ; resultat : nb_teams * 3 (1 sur l'evenement)
    # output : 3 (triplé ; pas forcément normalisé - [de proba])

    #[batch_size x input_size]RnnStdCell.py


    def __init__(self, nb_teams):
        self.__nb_teams = nb_teams
        self.non_trainable_params = {}
        self.non_trainable_params['away_transform'] = tf.Variable(initial_value=[[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]], trainable=False)
        self.non_trainable_params['inverted_away_transform'] = self.non_trainable_params['away_transform']
        self.trainable_param = {}
        self.trainable_param['b'] = tf.Variable(initial_value=[ 0.35,-0.25,-0.35], trainable=True)
        lnlambda = tf.Variable(initial_value=[-3.,-5.,-3.], trainable=True)
        self.trainable_param['lambda'] = tf.exp(lnlambda)

    @property
    def state_size(self):
        return tf.TensorShape([self.__nb_teams, 3])

    @property
    def output_size(self):
        return tf.TensorShape([3])

    def __call__(self, inputs, state, scope=None): # input: [batch, team, 3, 2], state: [batch, team, 3]
        #tf.slice(tensor,from, epaisseur)
        home = tf.slice(inputs, [0, 0, 0, 0], [-1, self.__nb_teams, 1, 1], name=None)  # [batch, team, 1, 1]
        home = tf.squeeze(home, axis=[3])                                              # [batch, team, 1]
        away = tf.slice(inputs, [0, 0, 1, 0], [-1, self.__nb_teams, 1, 1], name=None)  # [batch, team, 1, 1]
        away = tf.squeeze(away, axis=[3])                                              # [batch, team, 1]

        home_state = tf.batch_matmul(home, state, adj_x=True)   # [batch, 1, 3]
        home_state = tf.squeeze(home_state, axis=1)             # [batch, 3]
        away_state = tf.batch_matmul(away, state, adj_x=True)   # [batch, 1, 3]
        away_state = tf.squeeze(away_state, axis=1)             # [batch, 3]
        away_state = tf.matmul(away_state, self.non_trainable_params['away_transform'])  # [batch, 3] (self.non_trainable_params['reverse'] : [3,3])

        match_elo = home_state + away_state # [batch, 3]
        #map_fn -> for each sur le premier indice (iterable) de match_elo; apply function
        match_elo = tf.map_fn(lambda line: line + self.trainable_param['b'], match_elo) # [batch, 3] (self.trainable_param['b']: [3])

        output = tf.exp(match_elo) # [batch, 3]
        output = tf.map_fn(lambda line: line / tf.reduce_sum(line), output) # [batch, 3]

        result = tf.slice(inputs, [0, 0, 0, 1], [-1, 1, 3, 1], name=None) # [batch, 1, 3, 1]
        result = tf.squeeze(result, axis=[1,3])        # [batch, 3]

        loss = -tf.log(output +  1e-9) * result # [batch, 3]

        update = result - output                  # [batch, 3]
        update = tf.map_fn(lambda line: line*self.trainable_param['lambda'],update) # [batch, 3] (self.trainable_param['lambda'] : 3)
        update = tf.expand_dims(update, axis=1) # [batch, 1, 3]

        update_home = tf.batch_matmul(home, update)  # [batch, team, 3]
        update = tf.map_fn(lambda line: tf.matmul(line, self.non_trainable_params['inverted_away_transform']), update) # [batch, team, 3]
        update_away = tf.batch_matmul(away, update)  # [batch, team, 3]

        next_state = state + update_home + update_away # [batch, team, 3]

        return loss, next_state

#bashs=[[ [[[1,8],[2,8]]], [[[2,8],[2,8]]] , [[[9,8],[2,8]]] ],[ [[[3,8],[1,8]]] , [[[5,8],[2,8]]] , [[[6,8],[3,8]]] ]] # 2 bash * dim(bash) * 1 seule donnée
#bashs=[[ [1,2], [2,2] , [9,2] ],[ [3,1] , [5,2] , [6,3] ]]
#bashs=[[ [1], [2] , [2] ],[ [1] , [5] , [3] ]]

def create_list(dims, random=False):
    if dims == []:
        if random:
            return random.randint(1,9)
        else:
            return 0.
    mydims = copy.deepcopy(dims)
    dim = mydims.pop(0)
    return [create_list(mydims) for _ in range(dim)]

def get_bash():
    input_dim = [time_len,bashs_number,nb_teams,3,2]
    bashs = create_list(input_dim)

    for t in range(time_len):
        team_h = random.randint(0, nb_teams-1)
        team_a = random.randint(0, nb_teams-1)
        while team_a == team_h:
            team_a = random.randint(0, nb_teams - 1)
        result = random.randint(0, 2)
        print(team_h, team_a, result)
        bashs[t][0][team_h][0][0] = 1.
        bashs[t][0][team_a][1][0] = 1.
        bashs[t][0][0][result][1] = 1.

data = Data.Data(Params.FILE)

sess = tf.Session()
cell=BasicRNNCell(data.meta_datas["nb_teams"])
bashs = data.rnn_datas
inputs=tf.constant(bashs)

loss, state = tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32, time_major=True)


loss_total = tf.reduce_mean(tf.reduce_sum(loss,axis=[1,2]))
optimizer = tf.train.AdamOptimizer(0.1).minimize(loss_total)

final_state = tf.squeeze(state) # [time,team,3]


print('START!')
sess.run(tf.global_variables_initializer())

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
exit()


s = sess.run(state)

print(sess.run(loss_total))
print("lambda", sess.run(cell.trainable_param['lambda']))
print("b", sess.run( cell.trainable_param['b']))
sess.run(optimizer)


