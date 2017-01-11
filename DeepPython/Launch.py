# -*- coding: utf-8 -*-
import time
import csv
import math
from DeepPython import Params, Costs


class Launch:
    def __init__(self, data, model, bookmaker, type_slice, display=3):
        self.data = data
        self.model = model
        self.bookmaker = bookmaker
        self.type_slice = type_slice
        self.tf_operations = []
        self.session = None
        self.display = display

    def initialize(self):
        ll_res = Costs.Cost('logloss', feed_dict={'target': 'res', 'feature_dim': 1, 'regularized': False})
        rll_res = Costs.Cost('logloss', feed_dict={'target': 'res', 'feature_dim': 1, 'regularized': True})
        self.model.add_cost(ll_res, trainable=False)
        self.model.add_cost(rll_res, trainable=True)
        self.model.finish_init()
        self.model.set_params(Params.paramStd)

    def execute(self):
        if self.display <= 2:
            print("Evaluating model")

        self.initialize()

        ll_mean = 0.
        lls_mean = 0.

        self.data.init_slices(self.type_slice, feed_dict={'p_train': 0.5})
        now = time.strftime('%Y_%m_%d_%H_%M_%S')

        with open(Params.OUTPUT + '_' + now, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            nb_slices = self.data.nb_slices(self.type_slice) if self.type_slice != "Shuffle" else 10
            for i in range(nb_slices):
                t = time.time()
                self.model.reset()

                train_p = {'when_odd': False}
                test_p = {'when_odd': True}
                slice_train, slice_test = self.data.cget_both_slices(self.type_slice, train_p=train_p, test_p=test_p)

                if not (self.data.is_empty(slice_train) or self.data.is_empty(slice_test)):
                    if self.display <= 1:
                        print("Working on slice {} / {}".format(i+1, nb_slices))
                    if self.model.is_trainable():
                        self.set_current_slice(slice_train)
                        self.model.train('rll_res', Params.NB_LOOPS)

                    self.set_current_slice(slice_test)
                    ll = self.model.get_cost('ll_res')
                    prediction = self.model.run(self.model.prediction['res'])

                    keys = ['res', 'odds']
                    datas = {}
                    for key in keys:
                        datas[key] = slice_test[key]
                        if len(datas[key]) != len(prediction):
                            raise Exception('ERROR lengths do not match {} and {}'.format(len(datas[key]), len(prediction)))
                    for j in range(len(prediction)):
                        to_write = list(prediction[j]) + datas['odds'][j] + datas['res'][j]
                        if self.display <= 0:
                            print(to_write)
                        spamwriter.writerow(to_write)
                    ll_mean += ll
                    lls_mean += ll ** 2
                    if self.display <= 1:
                        print("Slice {}: cost {} computed in {} s.".format(i, str(ll)[:7], str(time.time() - t)[:4]))

        mean_total = ll_mean/self.data.nb_slices(self.type_slice)
        std_dev_total = math.sqrt(lls_mean / nb_slices - (ll_mean / nb_slices) ** 2)
        if self.display <= 2:
            print('Mean: {} / Std_dev: {}'.format(str(mean_total)[:7], str(std_dev_total)[:7]))

        self.model.close()

    def target_loss(self, params):
        self.model.session.run(self.model.init_all)
        self.model.set_params(params)
        s_train, s_test = self.data.get_both_slices('Shuffle', train_p={'when_odd': False}, test_p={'when_odd': True})
        self.set_current_slice(s_train)
        self.model.train('rll_res', Params.NB_LOOPS)
        self.set_current_slice(s_test)
        res = self.model.get_cost('ll_res')
        if math.isnan(res):
            res = float('inf')
        print(res, params)
        return res  # self.model.get_cost('ll_res')

    def grid_search(self):
        self.data.init_slices('Shuffle', feed_dict={'p_train': 0.5})
        fun = self.target_loss
        to_optimize = ['metaparamj2', 'metaparam2', 'bais_ext', 'draw_elo']
        metaparams = self.model.meta_params()
        reset = lambda: self.data.next_slice('Shuffle')
        optimizer = Metaopti.Metaopti(fun, metaparams, to_optimize, reset)
        optimizer.init_paramrange()
        while optimizer.to_optimize:
            print(optimizer.opti_step())
            print(optimizer.to_optimize)

    def set_current_slice(self, s): 
        feed_dict = {}
        for key in self.model.features_data():
            feed_dict[self.model.ph_current_slice[key]] = s[key]
        self.model.session.run(self.model.tf_assign_slice, feed_dict=feed_dict)
