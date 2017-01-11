from copy import deepcopy
import random
import Params


class Slice:
    def __init__(self, py_datas, group, feed_dict=None):

        if feed_dict is None:
            feed_dict = {}

        self.switcher = {
            'Lpack': {'init': self.__init_lpack,
                      'get_slice': self.__get_slice_lpack,
                      'next_slice': self.__next_slice_lpack,
                      'cget_slice': self.__get_current_slice_lpack},
            'Shuffle': {'init': self.__init_shuffle,
                        'get_slice': self.__get_slice_shuffle,
                        'next_slice': self.__shuffle_slice,
                        'cget_slice': self.__get_slice_shuffle}
        }

        if group not in self.switcher:
            print(group)
            raise('Unkown type ' + group)

        self.__group = group
        self.__parameters = feed_dict
        self.__slices = {}
        self.__nb_matches = len(py_datas)

        self.nb_slices = 0

        self.switcher[self.__group]['init'](py_datas)

        if Params.CHECK_SLICE_OVERLAP:
            self.__check_slices()

    def get_slice(self, feed_dict):
        return lambda l: self.switcher[self.__group]['get_slice'](l, feed_dict)

    def cget_slice(self, feed_dict):
        return lambda l: self.switcher[self.__group]['cget_slice'](l, feed_dict)

    def next_slice(self):
        return self.switcher[self.__group]['next_slice']()

    def __init_lpack(self, py_datas):
        self.__slices = {'train': [], 'test': []}
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
                current_slice_train['name'] = 'train_' + str(current_nb_slice + 1)
                current_nb_slice += 1
                if current_nb_slice > 0:
                    self.__slices['test'].append(deepcopy(current_slice_test))
                self.__slices['train'].append(deepcopy(current_slice_train))
                current_slice_test['left'] = nb_matchs + 1
            nb_matchs += 1
        self.nb_slices = min(len(self.__slices['test']), len(self.__slices['train']))
        self.__quick_check_slices(self.__group)
        self.__slices['i'] = 0

    def __get_slice_lpack(self, l, feed_dict):
        left = self.__slices[feed_dict['label']][feed_dict['index']]['left']
        right = self.__slices[feed_dict['label']][feed_dict['index']]['right']
        return l[left:right]

    def __get_current_slice_lpack(self, l, feed_dict):
        feed_dict['index'] = self.__slices['i']
        return self.__get_slice_lpack(l, feed_dict)

    def __next_slice_lpack(self):
        self.__slices['i'] += 1

    #  Shuffle
    def __init_shuffle(self, _):
        self.nb_slices = 1
        self.__slices['internal_seed'] = random.getrandbits(64)

    def __get_slice_shuffle(self, l, feed_dict):
        n = len(l)
        random.seed(a=self.__slices['internal_seed'])
        if feed_dict['label'] == 'train':
            return random.sample(l, n)[:int(n * self.__parameters['p_train'])]
        else:
            return random.sample(l, n)[int(n * self.__parameters['p_train']):]

    def __shuffle_slice(self):
        self.__slices['internal_seed'] = random.getrandbits(64)

    def check_slices(self, train_p, test_p):
        datas = range(self.__nb_matches)
        train = self.get_slice(train_p)(datas)
        test = self.get_slice(test_p)(datas)
        if set(train) & set(test):
            raise 'overlapping test and train'

    def __quick_check_slices(self, group):
        if group == 'Lpack':
            for i in range(self.nb_slices):
                if self.__slices['train'][i]['right'] <= self.__slices['test'][i]['left'] \
                                            or self.__slices['train'][i]['left'] >= self.__slices['test'][i]['right']:
                    for test_train in ['test', 'train']:
                        if self.__slices[test_train][i]['name'] != test_train + '_' + str(i):
                            raise Exception('Slice number {} error: expected {}, but name is {}'
                                            .format(i, test_train + '_' + str(i), self.__slices['train'][i]['name']))
                else:
                    raise Exception('Overlapping train and test')
