# -*- coding: utf-8 -*-
import time
import Params
import tensorflow as tf
import csv
import math
import Data


class Launch:
    def __init__(self, data, model, algogen=None):
        
        self.data = data
        self.model = model
        self.algogen = algogen
        self.tf_operations = [] 
        self.session = None

    def Go(self): 

        self.model.define_logloss(regularized=False, trainable=False)
        self.model.define_logloss(regularized=True, trainable=True)
        self.model.finish_init()
        
        self.model.set_params(Params.paramStd)
        ll_mean = 0.
        lls_mean = 0.
        
        self.data.init_slices('Lpack')

        with open(Params.OUTPUT, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(self.data.slices['Lpack'].nb_slices):
                t = time.time()
                self.model.reset()

                slice_train = self.data.get_slice('Lpack', feed_dict={'index': i, 'label': 'train'})
                slice_test = self.data.get_slice('Lpack', feed_dict={'index': i, 'label': 'test'})
                
                if not (self.data.is_empty(slice_train) or self.data.is_empty(slice_test)):
                    print(i)
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
                            print('ERROR')
                    for j in range(len(prediction)):
                        to_write = list(prediction[j]) + datas['odds'][j] + datas['res'][j]
                        print(to_write)
                        spamwriter.writerow(to_write)
                    ll_mean += ll
                    lls_mean += ll**2
                    print(i, ll, '(time: ' + str(time.time() - t) + ')')
        
        print('Mean: ', ll_mean/self.data.slices['Lpack'].nb_slices)
        print('Std_Dev: ', math.sqrt(lls_mean/self.data.slices['Lpack'].nb_slices - (ll_mean/self.data.slices['Lpack'].nb_slices)**2))

        self.model.close()

    def set_current_slice(self, s): 
        feed_dict = {}
        for key in self.model.features_data():
            feed_dict[self.model.ph_current_slice[key]] = s[key]
        self.model.session.run(self.model.tf_assign_slice, feed_dict=feed_dict)
        
