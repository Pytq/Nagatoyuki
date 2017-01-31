# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:32:06 2016

@author: mlegrain
"""
import tensorflow as tf


FILE = '../Datas/DataClean/SP1TotalNotCrossChecked.txt'
FILE_TEST = '../Datas/DataClean/E0TotalNotCrossChecked.txt'
FILE_TEST2 = '../Datas/DataClean/F1TotalNotCrossChecked.txt'
OUTPUT = '../Output/output_new'
NB_LOOPS = 60

CHECK_SLICE_OVERLAP = False
MAX_GOALS = 5

#UNUSED
DATA_TYPE = tf.float32
MAX_SEED = int(1e16)

#Relative to modelSTD
data2_nb_teams = 159
data2_nb_saisons = 14
data2_nb_max_journee = 38
data2_nb_journee = (data2_nb_saisons + 1) * data2_nb_max_journee

paramStd = {'metaparam2': 1.837745, 'metaparam0': -22.0, 'metaparamj0': 0.0, 'metaparamj2': 3.873245}

init_dict = {'metaparam2': 1.837745, 'metaparam0': -22.0, 'metaparamj0': 0.0, 'metaparamj2': 3.873245}

