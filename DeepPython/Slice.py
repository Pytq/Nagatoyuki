from copy import deepcopy
import random


class Slice:
    def __init__(self, py_datas, group, feed_dict={}):
        
        self.group = group
        self.parameters = feed_dict
        
        if group == 'Lpack':
            self.slices = {'train': [], 'test': []}
            current_journee = -1
            current_nb_slice = -1
            nb_matchs = 0
            current_slice_test = {'left': 0, 'right': 0}
            current_slice_train = {'left': 0, 'right': 0}
            for row in py_datas:
                if current_journee != row["journee"] and nb_matchs > 3800:
                    current_journee = row["journee"]
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
            self.nb_slices = min(len(self.slices['test']), len(self.slices['train']))
            self.check_slices(group)

        elif group == 'Shuffle':
            self.nb_slices = 1
            self.internal_seed = random.getrandbits(64)

        else:
            print('Unkown type ' + type)

    def check_slices(self, group):
        if group == 'Lpack':
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
            return lambda l: self.shuffle_list(l, self.parameters['p_train'], feed_dict['label'] == 'train')
        else:
            print('warning Slices.get_slices')
            
                
            

        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
