# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:32:06 2016

@author: mlegrain
"""
import tensorflow as tf

FILE = '../Datas/DataClean/ResultWithOdds2.txt'
OUTPUT = '../Output/output'
NB_LOOPS = 2

CHECK_SLICE_OVERLAP = False
MAX_GOALS = 9

# UNUSED
DATA_TYPE = tf.float32


paramStd = {
    'metaparamj2': 7.596735999999996,
    'metaparamj1': -9.9995,
    'metaparamj0': -0.6004999999999996,
    'metaparam2': 1.809571999999999,
    'metaparam1': -10.2545,
    'metaparam0': -11.799499999999998,
    'bais_ext': 0.5396204199999999,
    'draw_elo': -0.4084012199999998
}

# Relative to modelSTD
# data2_nb_teams = 159
# data2_nb_saisons = 14
# data2_nb_max_journee = 38
# data2_nb_journee = (data2_nb_saisons + 1) * data2_nb_max_journee
