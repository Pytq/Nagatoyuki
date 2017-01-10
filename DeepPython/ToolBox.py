import tensorflow as tf
import datetime
import random

ELOCONST = -1.


def nb_tf_op():
    return len(tf.get_default_graph().get_operations())


def kron(i_, j_):
    if j_ == i_:
        return 1.
    else:
        return 0.

def first_time(nb_times):
    first_time_python = [[0.] for i in range(nb_times)]
    first_time_python[0] = [1.]
    return tf.constant(first_time_python)


def make_vector(pos, size):
    res = [0.]*size
    res[pos] = 1.
    return res


def temp(i_, j_):
    if j_ == i_:
        return 1.
    elif j_ == i_-1:
        return -1.
    else:
        return 0.


def result(k):
    if k < 0:
        return 0.
    elif k == 0:
        return 0.5
    else:
        return 1.


def result_vect(k):
    if k < 0:
        return [0., 0., 1.]
    if k == 0:
        return [0., 1., 0.]
    else:
        return [1., 0., 0.]


def format_name(s):
    return s.lower().replace("-", "").replace(" ", "")


def date_to_number(s):
    return datetime.date(int(s[6:10]), int(s[3:5]), int(s[0:2])).toordinal()


def date_to_number_slash_and_format(s):
    if len(s) == 8:
        year = int("20"+s[6:8])
    else:
        year = int(s[6:10])
    month = int(s[3:5])
    day = int(s[0:2])
    return datetime.date(year, month, day).toordinal()


def gen_date_to_id(s, nb_times, max_time, min_time):
    date = date_to_number(s)
    return int(nb_times * float(date - min_time)/float(max_time - min_time + 1))


def get_raw_elo_cost(metaparam0, metaparam1, elo, nb_times):
    cost1 = tf.reduce_mean(tf.square(elo)) * metaparam0 * ELOCONST ** 2
    cost2 = tf.reduce_mean(tf.square(tf.matmul(elo, first_time(nb_times))))
    cost2 *= metaparam1 * ELOCONST ** 2
    if metaparam0 == 0:
        if metaparam1 == 0:
            return tf.constant(0.)
        else:
            return cost2
    else:
        if metaparam1 == 0:
            return cost1
        else:
            return cost1 + cost2


def timediff_gen(nb_times):
    return tf.constant([[kron(i, j)-kron(j, i-1) for j in range(nb_times - 1)] for i in range(nb_times)])


def get_timediff_elo_cost(metaparam, elo, nb_times):
    timediff = timediff_gen(nb_times)
    if nb_times > 1:
        cost_diffelo = tf.reduce_mean(tf.square(tf.matmul(elo, timediff)))
        cost_diffelo *= metaparam * ELOCONST ** 2
        return cost_diffelo
    else:
        raise ValueError('Cannot compute timediff with nb_times < 2')


def get_elomatch(team, time, elo):
    elomatch = tf.matmul(team, elo)
    elomatch = tf.reduce_sum(elomatch * time, reduction_indices=[1])
    return elomatch


def sample(l):
    return random.sample(l,len(l))

def last_vector(n):
    res = [0.] * n
    res[n-1] = 1.
    return tf.constant(res)


def win_vector(n):
    return tf.constant([[result(i-j) for j in range(n)] for i in range(n)])


def alter(i, j, param):
    return 0


def alter_proba(param):
    return tf.constant([[alter(i,j,param) for j in range(10)] for i in range(10)])
