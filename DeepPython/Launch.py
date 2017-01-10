# -*- coding: utf-8 -*-
import time
import Params
import csv
import math
import Costs
import Metaopti

class Launch:
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.tf_operations = []
        self.session = None

    def init_model(self):
        ll_res = Costs.Cost('logloss', feed_dict={'target': 'res', 'feature_dim': 1, 'regularized': False})
        rll_res = Costs.Cost('logloss', feed_dict={'target': 'res', 'feature_dim': 1, 'regularized': True})
        self.model.add_cost(ll_res, trainable=False)
        self.model.add_cost(rll_res, trainable=True)
        self.model.finish_init()

    def go(self):
        self.model.set_params(Params.paramStd)
        ll_mean = 0.
        lls_mean = 0.

        TYPE_SLICE = 'Shuffle'
        self.data.init_slices(TYPE_SLICE, feed_dict={'p_train': 0.5})

        with open(Params.OUTPUT, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            nb_iter = 10  # self.data.nb_slices(TYPE_SLICE))
            for i in range(nb_iter):
                self.data.shuffle_slice(TYPE_SLICE)
                t = time.time()
                self.model.reset()

                train_p = {'index': i, 'when_odd': False}
                test_p = {'index': i, 'when_odd': True}
                slice_train, slice_test = self.data.get_both_slices(TYPE_SLICE, train_p=train_p, test_p=test_p)
                # slice_train = self.data.get_slice(TYPE_SLICE, feed_dict=train_p)
                # slice_test = self.data.get_slice(TYPE_SLICE, feed_dict=test_p)

                if not (self.data.is_empty(slice_train) or self.data.is_empty(slice_test)):
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
                            raise 'ERROR'
                    for j in range(len(prediction)):
                        to_write = list(prediction[j]) + datas['odds'][j] + datas['res'][j]
                        # print(to_write)
                        spamwriter.writerow(to_write)
                    ll_mean += ll
                    lls_mean += ll**2
                    print("{0}/{1}: {2} (time: {3})".format(i+1, nb_iter, ll, time.time()-t))
        mean = ll_mean/nb_iter
        std_dev = math.sqrt(lls_mean/nb_iter - mean**2)
        print('Mean:    {}'.format(mean))
        print('Std_Dev: {}'.format(std_dev))

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
        reset = lambda: self.data.shuffle_slice('Shuffle')
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
