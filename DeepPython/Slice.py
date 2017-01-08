from copy import deepcopy
import random


class Slice:
    def __init__(self, py_datas, group, feed_dict={}):
        
        self.group = group
        # A changer et garder le dict !
        if 'p_train' in feed_dict:
            self.p_train = feed_dict['p_train']
        else:
            self.p_train = None
        if 'when_odd' in feed_dict:
            self.when_odd = feed_dict['when_odd']
        else:
            self.when_odd = None
        
        if group == 'Lpack':
            self.slices = {'train': [], 'test': []}
            current_journee = -1
            current_nb_slice = -1
            nb_matchs = 0
            current_slice_test = {'left': 0, 'right': 0}
            current_slice_train = {'left': 0, 'right': 0}
            for row in py_datas:
                if row:
                    [ID, saison, Date, journee, id1, id2, score1, score2, Spectateur, JourDeLaSemaine,
                     FTR, FTHG, FTAG, BbMxH, BbMxD, BbMxA, BbAvH, BbAvD, BbAvA, HTHG, HTAG, HTR, HS, AS, HST, AST,
                     HC, AC] = row
                    if current_journee != journee and nb_matchs > 3800:
                        current_journee = journee
                        current_slice_test['right'] = nb_matchs + 1
                        current_slice_test['name'] = 'test_' + str(current_nb_slice)
                        current_slice_train['right'] = nb_matchs + 1
                        current_slice_train['name'] = 'train_' + str(current_nb_slice+1)
                        current_nb_slice += 1
                        if current_nb_slice > 0:
                            self.slices['test'].append(deepcopy(current_slice_test))
                        self.slices['train'].append(deepcopy(current_slice_train))
                        current_slice_test['left'] = nb_matchs + 1
                    nb_matchs += 1
            self.check_slices(group)

        elif group == 'Shuffle':
            # self.slices = {'train': [], 'test': []}
            # p = 0.5
            # self.slices['test'] = [{'left': 0, 'right': int(self.data.nb_matchs * p)}]
            # self.slices['train'] = [{'left': int(self.data.nb_matchs * p)+1, 'right': self.data.nb_matchs}]
            self.nb_slices = 1
            self.internal_seed = random.getrandbits(64)

        else:
            print('Unkown type ' + type)

    def check_slices(self, group):
        if group == 'Lpack':
            self.nb_slices = min(len(self.slices['test']), len(self.slices['train']))
            for i in range(self.nb_slices):
                if self.slices['train'][i]['right'] <= self.slices['test'][i]['left'] or self.slices['train'][i]['left'] >= self.slices['test'][i]['right']:
                    if self.slices['train'][i]['name'] != 'train_'+str(i) or self.slices['test'][i]['name'] != 'test_'+str(i):
                        print('slice error')
                else:
                    print('overlapping train and test')

    def shuffle_list(self, l, p, train):
        n = len(l)
        random.seed(a=self.internal_seed)
        if train:
            return random.sample(l, n)[:int(n*p)]
        else:
            return random.sample(l, n)[int(n*p):]

    def shuffle_slice(self):
        self.internal_seed = random.getrandbits(64)

    def get_slice(self, feed_dict):
        if self.group in ['Lpack']:
            return lambda l: l[self.slices[feed_dict['label']][feed_dict['index']]['left']:self.slices[feed_dict['label']][feed_dict['index']]['right']]
        elif self.group == 'Shuffle':
            return lambda l: self.shuffle_list(l, self.p_train, feed_dict['label'] == 'train')
        else:
            print('warning Slices.get_slices')
            
                
            

        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
