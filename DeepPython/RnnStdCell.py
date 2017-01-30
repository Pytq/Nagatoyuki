import tensorflow as tf

class BasicRNNCell(tf.nn.rnn_cell.RNNCell):
    #memoryState : nb_teams * 3 (elo win ; elo tie ; elo lose)
    #inputs * 2 : match (nb_teams * 2 (1,1)) ; resultat : nb_teams * 3 (1 sur l'evenement)
    #output : 3 (triplé ; pas forcément normalisé - [de proba])

    #[batch_size x input_size]RnnStdCell.py

    def __init__(self):
        pass

    @property
    def state_size(self):
        return tf.TensorShape(3)

    @property
    def output_size(self):
        return tf.TensorShape(1)

    def __call__(self, inputs, state, scope=None):
        x = inputs[0]
        y = inputs[1]
        next_state = x + state
        output = tf.expand_dims(tf.reduce_sum(x, axis=1), axis=1)
        return output, next_state

sess = tf.Session()
cell=BasicRNNCell()
in1 = [[[[1.,2.,3.]],[[4.,5.,6.]]], [[[2.,2.]],[[0.,8.]]]]

inputs= [tf.constant(v) for v in in1]
inputsf = [inputs[0], inputs[1]]

output, state = tf.nn.dynamic_rnn(cell,inputsf,dtype=tf.float32, time_major=True)

print('output: {}'.format(sess.run(output)))
print('state: {}'.format(sess.run(state)))
