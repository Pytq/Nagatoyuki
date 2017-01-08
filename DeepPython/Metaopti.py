import random
from scipy.optimize import differential_evolution
import numpy as np
import math
import time

class MetaOptiDiffEvo:
    def __init__(self, data, paramnames, bounds, model):
        self.data = data
        self.paramnames = paramnames
        self.bounds = bounds # bounds = [(-12., 2.), (-12., 2.), (-6., 8.), (-0.55, -0.2), (-1., 2.)]
        self.model = model
        self.t = time.time()

        if len(paramnames) != len(bounds):
            raise 'paramnames and bounds must match their respective size'

    def to_optimise(self, x):
        paramafter = {}
        for i in range(len(self.paramnames)):
            paramafter[self.paramnames[i]] = x[i]
        self.data.get(0.5)
        self.model.set_train_data(self.data.train)
        self.model.set_test_data(self.data.test)
        self.model.reset()

        self.model.set_params(paramafter)
        for i in range(100):
            self.model.train_res()
        cost = self.model.get_cost('test')
        for key in paramafter:
            if key !=  'bais_ext':
                paramafter[key] = math.exp(paramafter[key])
            paramafter[key] = round(paramafter[key], 6)
        print int(round(10000*cost)), int(10*(time.time() - self.t)), paramafter
        self.t = time.time()
        return cost

    def launch(self):
        result = differential_evolution(self.to_optimise, self.bounds)
        return result.x, result.fun



class MetaOpti:
    def __init__(self, data, paramrange, param, model, target, M, L):
        self.data = data
        self.paramrange = paramrange
        self.param = param
        self.model = model
        self.target = target
        self.M = M
        self.L = L

    def mongo_opti(self):
        key = random.choice(self.paramrange.keys())
        cost = 200000.
        costmax = 0.
        elparam = None
        m, p, t = self.paramrange[key]
        if t == 'exp':
            rangee = [m * p**(i-self.M/2) for i in range(self.M)]
        elif t == 'lin':
            rangee = [m + p * (i - self.M / 2) for i in range(self.M)]
        else:
            print 'WTF ???? (line 22 METAOPTI LOL)'
        for x in rangee:
            self.param[key] = x
            self.model.reset()
            self.model.set_params(self.param)
            for i in range(100):
                self.model.train_res()
            temp = self.target()
            print key, x, 'current_los', temp
            if temp < cost:
                cost = temp
                elparam = x
            if temp > costmax:
                costmax = temp
        self.param[key] = elparam
        if t == 'exp':
            self.paramrange[key] = elparam, p**self.L, t
        else:
            self.paramrange[key] = elparam, p * self.L, t
        print 'final', cost,
        if costmax - cost < 0.001:
            del self.paramrange[key]



