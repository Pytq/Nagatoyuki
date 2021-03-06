# -*- coding: utf-8 -*-
import time
import csv
import math

import itertools

from DeepPython import Params, Costs, Metaopti
import tensorflow as tf


class Launch:
    def __init__(self, data, dict_models, type_slice, display=3, init_dict=None):
        self.init_dict = init_dict
        self.data = data
        self.models = dict_models
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
                model.add_cost(rll_res, trainable=True) #model.add_cost(rll_res, trainable=True)
            model.finish_init()
            model.set_params(Params.paramStd)

    def execute(self):
        if self.display <= 2:
            print("Evaluating Models: {}".format(list(self.models)))
        print()

        diff_names = [{"from": x1, "to": x2, "name": "zDiff_" + x1 + "_" + x2} for x1, x2 in itertools.product(self.models, self.models) if x1 < x2]
        evaluation = {x: {"prediction": None, "ll_sum": 0, "lls_sum": 0, "current_ll": None, "time": 0} for x in list(self.models)
                                            + [v["name"] for v in diff_names]}
        for v in diff_names:
            for t in ["from", "to"]:
                evaluation[v["name"]][t] = v[t]

        self.data.init_slices(self.type_slice, feed_dict={'p_train': 0.5})
        timeNow = time.strftime('%Y_%m_%d_%H_%M_%S')

        with open(Params.OUTPUT + '_' + timeNow, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            nb_slices = self.data.nb_slices(self.type_slice) if self.type_slice != "Shuffle" else 4
            train_p = {'when_odd': True}
            test_p = {'when_odd': True}
            for i in range(nb_slices):
                if i % 2 == 1 and self.type_slice == "Shuffle" and False:
                    slice_test, slice_train = self.data.cget_both_slices(self.type_slice, train_p=test_p, test_p=train_p)
                    if self.display <= 1:
                        print("Shuffle mode: test and train are swapped")
                else:
                    slice_train, slice_test = self.data.cget_both_slices(self.type_slice, train_p=train_p, test_p=test_p)

                if not (self.data.is_empty(slice_train) or self.data.is_empty(slice_test)):
                    if self.display <= 1:
                        print("Working on slice {} / {} ...".format(i + 1, nb_slices))

                    for model_name, model in self.models.items():
                        timeStartModel = time.time()
                        model.reset_trainable_parameters()

                        self.set_current_slice(slice_train, model_name)
                        rll_train = 0
                        if model.is_trainable():
                            model.train('rll_res', Params.NB_LOOPS)
                            rll_train = model.get_cost('rll_res')
                            print(" R ZZ " + str(model_name) + " lossètrainR : " + str(rll_train))
                        ll_train = model.get_cost('ll_res')
                        print(" ZZ " + str(model_name) + " lossètrain : " + str(ll_train))

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
                        evaluation[model_name]["current_ll"] = ll
                        evaluation[model_name]["train_ll"] = ll_train
                        evaluation[model_name]["train_rll"] = rll_train
                        evaluation[model_name]["time"] = time.time()-timeStartModel
                self.data.next_slice(self.type_slice)
                for key, val in sorted(evaluation.items()):
                    if "zDiff" in key:
                        val["current_ll"] = evaluation[val["from"]]["current_ll"] - evaluation[val["to"]]["current_ll"]
                        val["train_ll"] = evaluation[val["from"]]["train_ll"] - evaluation[val["to"]]["train_ll"]
                        val["train_rll"] = evaluation[val["from"]]["train_rll"] - evaluation[val["to"]]["train_rll"]
                        to_write = [key, str(val["current_ll"]), str(val["train_ll"]), str(val['train_rll'])]
                        print(to_write)
                        spamwriter.writerow(to_write)
                    val["ll_sum"] += val["current_ll"]
                    val["lls_sum"] += val["current_ll"] ** 2
                    print("-- {} - {} : MEAN : {} DEV : {}.".format(key, i + 1,
                                                                            str(val["ll_sum"]/(i+1))[:7],
                                                                            str(val["lls_sum"]/(i+1))[:7]))
                    if self.display <= 1:
                        print("{} - Slice {}: Cost {} computed in {} s.".format(key, i+1,
                                                                                str(val["current_ll"])[:7],
                                                                                str(val["time"])[:4]))
                if 'Regresseur' in self.models:
                    for x in self.models['Regresseur'].trainable_params:
                        if 'alpha' in x:
                            print(x, self.models['Regresseur'].run(self.models['Regresseur'].trainable_params[x]))
                print()

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
        #model.session.run(model.)
        #model.reset_trainable_parameters()
        model.set_params(params)
        s_train, s_test = self.data.get_both_slices('Shuffle', train_p={'when_odd': False}, test_p={'when_odd': True})
        self.set_current_slice(s_train, model_name)
        model.train('rll_res', Params.NB_LOOPS)
        self.set_current_slice(s_test, model_name)
        res = model.get_cost('ll_res')
        if math.isnan(res):
            res = float('inf')
        print(model_name, res, params)
        return res  # self.model.get_cost('ll_res')

    def grid_search(self, model_name):
        model = self.models[model_name]
        model.reset_trainable_parameters()
        self.data.init_slices('Shuffle', feed_dict={'p_train': 0.5})
        fun = lambda params: self.target_loss(params, model_name)
        to_optimize = model.meta_params()
        metaparams = model.meta_params()
        reset = lambda: self.data.next_slice('Shuffle')
        optimizer = Metaopti.Metaopti(fun, metaparams, to_optimize, reset, init_dict=self.init_dict)
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
