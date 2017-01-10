# -*- coding: utf-8 -*-
import time
import Params
import csv
import math
import Costs


class Launch:
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.tf_operations = []
        self.session = None

    def Go(self):
        ll_res = Costs.Cost('logloss', feed_dict={'target': 'res', 'feature_dim': 1, 'regularized': False})
        rll_res = Costs.Cost('logloss', feed_dict={'target': 'res', 'feature_dim': 1, 'regularized': True})
        self.model.add_cost(ll_res, trainable=False)
        self.model.add_cost(rll_res, trainable=True)
        self.model.finish_init()

        # self.model.set_params(Params.paramStd)
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

                slice_train = self.data.get_slice(TYPE_SLICE, feed_dict={'index': i, 'label': 'train', 'when_odd': False})
                slice_test = self.data.get_slice(TYPE_SLICE, feed_dict={'index': i, 'label': 'test', 'when_odd': True})
                
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

    def set_current_slice(self, s): 
        feed_dict = {}
        for key in self.model.features_data():
            feed_dict[self.model.ph_current_slice[key]] = s[key]
        self.model.session.run(self.model.tf_assign_slice, feed_dict=feed_dict)
