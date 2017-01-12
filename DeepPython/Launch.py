# -*- coding: utf-8 -*-
import time
import csv
import math
from DeepPython import Params, Costs, Metaopti


class Launch:
    def __init__(self, data, model, bookmaker, type_slice, display=3):
        self.data = data
        self.model = model
        self.bookmaker = bookmaker
        self.names = {"Model": self.model}
        if self.bookmaker is not None:
            self.names["Bookmaker"] = self.bookmaker
        self.type_slice = type_slice
        self.tf_operations = []
        self.session = None
        self.display = display
        self.initialize()

    def initialize(self):
        ll_res = Costs.Cost('logloss', feed_dict={'target': 'res', 'feature_dim': 1, 'regularized': False})
        rll_res = Costs.Cost('logloss', feed_dict={'target': 'res', 'feature_dim': 1, 'regularized': True})
        for element in self.names.values():
            element.add_cost(ll_res, trainable=False)
            element.add_cost(rll_res, trainable=True)
            element.finish_init()
            element.set_params(Params.paramStd)

    def execute(self):
        if self.display <= 2:
            print("Evaluating Model vs Bookmaker")

        evaluation = {x: {"prediction": None, "ll_mean": 0, "lls_mean": 0} for x in self.names}

        self.data.init_slices(self.type_slice, feed_dict={'p_train': 0.5})
        now = time.strftime('%Y_%m_%d_%H_%M_%S')

        with open(Params.OUTPUT + '_' + now, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            nb_slices = self.data.nb_slices(self.type_slice) if self.type_slice != "Shuffle" else 3
            for i in range(nb_slices):
                t = time.time()
                train_p = {'when_odd': False}
                test_p = {'when_odd': True}
                slice_train, slice_test = self.data.cget_both_slices(self.type_slice, train_p=train_p, test_p=test_p)

                if not (self.data.is_empty(slice_train) or self.data.is_empty(slice_test)):
                    if self.display <= 1:
                        print("Working on slice {} / {}".format(i + 1, nb_slices))
                    for element_name, element in self.names.items():
                        element.reset()
                        if element.is_trainable():
                            self.set_current_slice(slice_train, element_name)
                            element.train('rll_res', Params.NB_LOOPS)

                        self.set_current_slice(slice_test, element_name)

                        ll = element.get_cost('ll_res')
                        prediction = element.run(element.prediction['res'])

                        datas = {}
                        for key in ['res', 'odds']:
                            datas[key] = slice_test[key]
                            if len(datas[key]) != len(prediction):
                                raise Exception(
                                    'Lengths do not match {} and {}'.format(len(datas[key]), len(prediction)))
                        for j in range(len(prediction)):
                            to_write = list(prediction[j]) + datas['odds'][j] + datas['res'][j]
                            if self.display <= 0:
                                print(to_write)
                            spamwriter.writerow(to_write)
                        evaluation[element_name]["ll_mean"] += ll
                        evaluation[element_name]["lls_mean"] += ll ** 2
                        if self.display <= 1:
                            print(
                                "{} - Slice {}: Cost {} computed in {} s.".format(element_name, i, str(ll)[:7], str(time.time() - t)[:4]))

        for element_name in self.names:
            evaluation[element_name]["Mean"] = evaluation[element_name]["ll_mean"] / nb_slices
            evaluation[element_name]["StdDev"] = math.sqrt(evaluation[element_name]["lls_mean"] / nb_slices
                                                           - (evaluation[element_name]["ll_mean"] / nb_slices) ** 2)
            if self.display <= 2:
                print('{} - Mean: {} / Std_dev: {}'.format(element_name, str(evaluation[element_name]["Mean"])[:7],
                                                           str(evaluation[element_name]["StdDev"])[:7]))

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
        reset = self.data.next_slice('Shuffle')
        optimizer = Metaopti.Metaopti(fun, metaparams, to_optimize, reset)
        optimizer.init_paramrange()
        while optimizer.to_optimize:
            print(optimizer.opti_step())
            print(optimizer.to_optimize)

    def set_current_slice(self, s, elem_name):
        feed_dict = {}
        elem = self.names[elem_name]
        for key in elem.features_data():
            feed_dict[elem.ph_current_slice[key]] = s[key]
        elem.session.run(elem.tf_assign_slice, feed_dict=feed_dict)
