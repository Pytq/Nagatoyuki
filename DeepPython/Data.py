import csv
import ToolBox
import copy
import Params
from random import shuffle

import Slice
ALL_FEATURES = ['saison', 'team_h', 'team_a', 'res', 'score_h', 'score_a', 'journee',
                'odd_win_h', 'odd_tie', 'odd_los_h', 'odds']

                
class Data:
    @staticmethod
    def check_len(s):
        check_len = []
        for key in s:
            check_len.append(len(s[key]))
        if check_len[1:] != check_len[:-1]:
            raise 'Data.py check_len'

    @staticmethod
    def is_empty(s):
        return s[list(s.keys())[0]] == []

    def __init__(self, filename, features=ALL_FEATURES, remove_features=[], typeslices='overtime'):

        self.filename = filename
        self.features = [x for x in features if x not in remove_features]
        
        self.datas = {}
        for key in features:
            self.datas[key] = []
        self.shuffled_datas = {}

        self.slices = {}

        self.nb_matchs = 0
        self.py_datas = []
        
        self.dict_dataToModel= {"nb_teams" : 159,
                                "nb_saisons" : 14,
                                "nb_max_journee" : 38}
                                
        self.dict_dataToModel["nb_journee"]=int((self.dict_dataToModel["nb_saisons"]+1) * self.dict_dataToModel["nb_max_journee"])
        
        self.get_matches()

        Data.check_len(self.datas)


##### Getting datas from .txt : 

    def get_matches(self):
        with open(self.filename) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            spamreader.__next__()
            for row in spamreader:
                if row:
                    [ID, saison, Date, journee, id1, id2, score1, score2, Spectateur, JourDeLaSemaine,
                     FTR, FTHG, FTAG, BbMxH, BbMxD, BbMxA, BbAvH, BbAvD, BbAvA, HTHG, HTAG, HTR, HS, AS, HST, AST,
                     HC, AC] = row
                    self.nb_matchs += 1
                    self.append_match(int(id1), int(id2), int(score1),
                                      int(score2), int(saison) - 2003, int(journee), BbMxH, BbMxD, BbMxA)
                    self.py_datas.append(row)

    def append_match(self, id1, id2, score1, score2, saison, journee, win_odd, tie_odd, los_odd):
        if win_odd == '':
            win_odd = -1.
            tie_odd = -1.
            los_odd = -1.
        self.datas['odd_win_h'].append([float(win_odd)])
        self.datas['odd_tie'].append([float(tie_odd)])
        self.datas['odd_los_h'].append([float(los_odd)])
        self.datas['odds'].append([float(win_odd), float(tie_odd), float(los_odd)])
        self.datas['saison'].append(ToolBox.make_vector(saison, Params.data2_nb_saisons))
        self.datas['team_h'].append(ToolBox.make_vector(id1, Params.data2_nb_teams))
        self.datas['team_a'].append(ToolBox.make_vector(id2, Params.data2_nb_teams))
        score_team_h = min(int(score1), 9)
        score_team_a = min(int(score2), 9)
        self.datas['score_h'].append(ToolBox.make_vector(score_team_h, 10))
        self.datas['score_a'].append(ToolBox.make_vector(score_team_a, 10))
        self.datas['res'].append(ToolBox.result_vect(int(score1) - int(score2)))
        self.datas['journee'].append(ToolBox.make_vector(journee + Params.data2_nb_max_journee*saison, Params.data2_nb_journee))

    def shuffle_datas(self):
        shuffle(self.shuffled_datas)

##### Gestion des slices:

    def init_slices(self, group, feed_dict={}):
        self.slices[group] = Slice.Slice(self.py_datas, group, feed_dict=feed_dict)
   
    def get_slice(self, group, feed_dict={}):
        if group not in self.slices:
            self.init_slices(group)
        extract_slice = self.slices[group].get_slice(feed_dict)
        s = {}
        for key in self.datas:
            s[key] = extract_slice(self.datas[key])
        Data.check_len(s)

        if feed_dict['when_odd']:
            s2 = {}
            for key in s:
                s2[key] = []
            for i in range(len(s['odd_tie'])):
                if s['odd_tie'][i][0] >= 0.:
                    for key in s2:
                        s2[key].append(s[key][i])
            s = s2

        return s

        

        

#    def get_elos(self, countries=None, times='all'):
#        if countries is None:
#            countries = [self.id_to_country[i] for i in range(Params.data2_nb_teams)]
#        else:
#            countries = map(ToolBox.format_name, countries)
#        if times == 'all':
#            times = range(Params.data2_nb_saisons)
#        elif times == 'last':
#            times = [Params.data2_nb_saisons - 1]
#        elos = {}
#        for country in countries:
#            elos[country] = [200*self.elo[self.country_to_id[country]][t] for t in times]
#        return elos

        
             #seed = random.randint(0, MAX_SEED)
        #for key in self.features:
        #    self.tf_datas[key] = tf.constant(self.datas[key], dtype=DATA_TYPE)  # Essayer avec variable trainable=False
        #    self.tf_shuffled_datas[key] = tf.random_shuffle(self.tf_datas[key], seed=seed)

#    def add_slices(self, p=None):
#        if p is not None:
#            slices = [{'left': 0, 'right': int(self.nb_matchs * p), 'name': 'train', 'shuffled': True},
#                      {'left': int(self.nb_matchs * p), 'right': -1, 'name': 'test', 'shuffled': True}]
#            map(self.create_tf_slice, slices)
#        if p is None:
#            for s in self.timeslices:
#                self.create_tf_slice(s)
#        print('Slices created. ' + str(ToolBox.nb_tf_op()) + ' nodes in tf.Graph')
#
#    def create_tf_slice(self, s):
#        self.tf_slices[s['name']] = {}
#        for key in self.features:
#            if s['shuffled']:
#                shuffled_slice = tf.slice(self.tf_shuffled_datas[key], [s['left'], 0], [s['right'], -1])
#                self.tf_slices[s['name']][key] = tf.Variable(0., validate_shape=False,
#                                                             dtype=DATA_TYPE, trainable=False, collections=[])
#                self.shuffle.append(tf.assign(self.tf_slices[s['name']][key], shuffled_slice, validate_shape=False))
#            else:
#                value = tf.slice(self.tf_datas[key], [s['left'], 0], [s['right'], -1])
#                self.tf_slices[s['name']][key] = value # tf.Variable(value, dtype=DATA_TYPE, trainable=False, collections=[])
