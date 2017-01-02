from copy import deepcopy

class Slice:

    def __init__(self, py_datas, group):
        
        self.group = group
        
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
            self.slices = {'train': [], 'test': []}
            p = 0.5
            self.slices['test'] = [{'left': 0, 'right': int(self.data.nb_matchs * p)}]
            self.slices['train'] = [{'left': int(self.data.nb_matchs * p)+1, 'right': self.data.nb_matchs}]
            self.nb_slices = 1

        else:
            print('Unkown type ' + type)
            
            
    def check_slices(self, group):
        self.nb_slices = min(len(self.slices['test']), len(self.slices['train']))
        for i in range(self.nb_slices):
            if self.slices['train'][i]['right'] <= self.slices['test'][i]['left'] or self.slices['train'][i]['left'] >= self.slices['test'][i]['right']:
                if self.slices['train'][i]['name'] != 'train_'+str(i) or self.slices['test'][i]['name'] != 'test_'+str(i):
                    print('slice error')
            else:
                print('overlapping train and test')
    
    def get_slice(self, index=0, label=None):
        if self.group in ['Lpack', 'Shuffle']:
            return lambda l: l[self.slices[label][index]['left']:self.slices[label][index]['right']]
        else:
            print('warning Slices.get_slices')

    def is_shuffled(self):
        if self.group in ['Lpack']:
            return False
        else:
            return True
            
                
            

        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
