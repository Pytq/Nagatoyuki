# -*- coding: utf-8 -*-
import time
import csv
import math
from DeepPython import Params, Costs, Metaopti
import tensorflow as tf


class Launch:
    def __init__(self, data, dictModels, type_slice, display=3):
        self.data = data
        self.models = dictModels
        self.type_slice = type_slice
        self.tf_operations = []
        self.session = None
        self.display = display
        
        self.initialize()

    def initialize(self):
        ll_res = Costs.Cost('logloss', feed_dict={'target': 'res', 'feature_dim': 1, 'regularized': False})
        rll_res = Costs.Cost('logloss', feed_dict={'target': 'res', 'feature_dim': 1, 'regularized': True})
        
        for model in self.models.values():
            model.add_cost(ll_res, trainable=False)
            if model.is_trainable():
                model.add_cost(rll_res, trainable=True)
            model.finish_init()
            model.set_params(Params.paramStd)

    def execute(self):
        if self.display <= 2:
            print("Evaluating Models: {}".format(list(self.models)))
        print()

        evaluation = {x: {"prediction": None, "ll_sum": 0, "lls_sum": 0} for x in list(self.models) + ["Diff"]}

        self.data.init_slices(self.type_slice, feed_dict={'p_train': 0.5})
        timeNow = time.strftime('%Y_%m_%d_%H_%M_%S')

        with open(Params.OUTPUT + '_' + timeNow, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            nb_slices = self.data.nb_slices(self.type_slice) if self.type_slice != "Shuffle" else 2
            train_p = {'when_odd': False}
            test_p = {'when_odd': True}
            for i in range(nb_slices):
                if i == 1 and self.type_slice == "Shuffle":
                    slice_train, slice_test = slice_test, slice_train
                    if self.display <= 1:
                        print("Shuffle mode: test and train are swapped")
                else:
                    slice_train, slice_test = self.data.cget_both_slices(self.type_slice, train_p=train_p, test_p=test_p)

                if not (self.data.is_empty(slice_train) or self.data.is_empty(slice_test)):
                    if self.display <= 1:
                        print("Working on slice {} / {} ...".format(i + 1, nb_slices))

                    for model_name, model in self.models.items():
                        timeStartModel = time.time()
                        model.reset()

                        if model.is_trainable():
                            self.set_current_slice(slice_train, model_name)
                            model.train('rll_res', Params.NB_LOOPS)

                        self.set_current_slice(slice_test, model_name)

                        ll = model.get_cost('ll_res')

                        prediction = model.run(model.prediction['res'])

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
                        evaluation[model_name]["current_ll"] = ll
                        evaluation[model_name]["time"] = time.time()-timeStartModel
                self.data.next_slice(self.type_slice)
                evaluation["Diff"]["current_ll"] = \
                    evaluation["Bookmaker"]["current_ll"] - evaluation["EloStd"]["current_ll"]
                evaluation["Diff"]["time"] = 0.
                for key in evaluation.keys():
                    evaluation[key]["ll_sum"] += evaluation[key]["current_ll"]
                    evaluation[key]["lls_sum"] += evaluation[key]["current_ll"] ** 2
                    if self.display <= 1:
                        print("{} - Slice {}: Cost {} computed in {} s.".format(key, i+1,
                                                                                str(evaluation[key]["current_ll"])[:7],
                                                                                str(evaluation[key]["time"])[:4]))
                print()
            for model in self.models.values():
                model.close()

        for model_name in evaluation.keys():
            evaluation[model_name]["Mean"] = evaluation[model_name]["ll_sum"] / nb_slices
            evaluation[model_name]["StdDev"] = math.sqrt(evaluation[model_name]["lls_sum"] / nb_slices
                                                           - evaluation[model_name]["Mean"] ** 2)
            if nb_slices > 1:
                evaluation[model_name]["StdDev"] /= math.sqrt(nb_slices - 1)
            if self.display <= 2:
                print('{} - Mean: {}% / Std_dev: {}'.format(model_name,
                                                            str(100*evaluation[model_name]["Mean"])[:7],
                                                            str(100*evaluation[model_name]["StdDev"])[:7]))

    def target_loss(self, params, model_name):
        model = self.models[model_name]
        model.session.run(model.init_all)
        model.set_params(params)
        s_train, s_test = self.data.get_both_slices('Shuffle', train_p={'when_odd': False}, test_p={'when_odd': True})
        self.set_current_slice(s_train)
        model.train('rll_res', Params.NB_LOOPS)
        self.set_current_slice(s_test, model_name)
        res = model.get_cost('ll_res')
        if math.isnan(res):
            res = float('inf')
        print(model_name, res, params)
        return res  # self.model.get_cost('ll_res')

    def grid_search(self, model_name):
        model = self.models[model_name]
        self.data.init_slices('Shuffle', feed_dict={'p_train': 0.5})
        fun = lambda params: self.target_loss(params, model_name)
        to_optimize = ['metaparamj2', 'metaparam2', 'bais_ext', 'draw_elo']
        metaparams = model.meta_params()
        reset = lambda: self.data.next_slice('Shuffle')
        optimizer = Metaopti.Metaopti(fun, metaparams, to_optimize, reset)
        optimizer.init_paramrange()
        while optimizer.to_optimize:
            print(optimizer.opti_step())
            print(optimizer.to_optimize)

    def set_current_slice(self, s, model_name):
        feed_dict = {}
        model = self.models[model_name]
        for key in model.features_data():
            feed_dict[model.ph_current_slice[key]] = s[key]
        model.session.run(model.tf_assign_slice, feed_dict=feed_dict)
