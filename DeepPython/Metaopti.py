import random
from copy import copy
import ToolBox


class Metaopti:
    def __init__(self, fun, params, reset, customizator=None):
        if customizator is None:
            customizator = {}
        self.__fun = fun
        self.__params = {}
        self.to_optimize = params
        self.__paramrange = {}
        for key in params:
            self.__params[key] = 0.
            self.__paramrange[key] = 0.5
        self.__reset = reset
        self.__custumizator = {'k': 5, 'alpha': 0.8, 'treshold': 0.005}
        for key in customizator:
            self.__custumizator[key] = customizator[key]

    def init_paramrange(self):
        self.__reset()
        x = copy(self.__params)
        for key in self.__params:
            print(key)
            prange = 0
            for sign in [-1., 1.]:
                x[key] = 0.
                i = float("inf")
                j = float("inf")
                z = self.__fun(x)
                while (z + self.__custumizator['treshold'] < j or j + self.__custumizator['treshold'] < i) and abs(x[key]) < 10**8:
                    i = j
                    j = z
                    x[key] = x[key] * 2. + sign
                    z = self.__fun(x)
                if x[key] >= 10**8:
                    raise 'out of range'
                prange = max(prange, abs(x[key]))
            self.__paramrange[key] = prange
            x[key] = 0.

    def map_paramrange(self, mapper):
        for key in self.__paramrange:
            self.__paramrange[key] = mapper(self.__paramrange[key])

    def opti_step(self):
        x = copy(self.__params)
        keys = ToolBox.sample(self.to_optimize)
        self.__reset()
        start_value = self.__fun(self.__params)
        while keys:
            key = keys.pop()
            y_min = float("inf")
            y_max = -float("inf")
            x_min = x[key]
            for k in range(self.__custumizator['k']):
                x[key] = self.__params[key] + self.__paramrange[key] * (float(k) / (self.__custumizator['k'] - 1.) - 0.5)
                y = self.__fun(x)
                if y < y_min:
                    y_min = y
                    x_min = x[key]
                if y > y_max:
                    y_max = y
            self.__paramrange[key] *= self.__custumizator['alpha']
            self.__params[key] = x_min
            x[key] = x_min
            if y_max - y_min < self.__custumizator['treshold']:
                self.to_optimize.remove(key)
        current_value = self.__fun(self.__params)
        return start_value - current_value, current_value, start_value


# class MetaOptiDiffEvo:
#
#     def __init__(self, data, paramnames, bounds, model):
#         self.data = data
#         self.paramnames = paramnames
#         self.bounds = bounds # bounds = [(-12., 2.), (-12., 2.), (-6., 8.), (-0.55, -0.2), (-1., 2.)]
#         self.model = model
#         self.t = time.time()
#
#         if len(paramnames) != len(bounds):
#             raise 'paramnames and bounds must match their respective size'
#
#     def to_optimise(self, x):
#         paramafter = {}
#         for i in range(len(self.paramnames)):
#             paramafter[self.paramnames[i]] = x[i]
#         self.data.get(0.5)
#         self.model.set_train_data(self.data.train)
#         self.model.set_test_data(self.data.test)
#         self.model.reset()
#
#         self.model.set_params(paramafter)
#         for i in range(100):
#             self.model.train_res()
#         cost = self.model.get_cost('test')
#         for key in paramafter:
#             if key !=  'bais_ext':
#                 paramafter[key] = math.exp(paramafter[key])
#             paramafter[key] = round(paramafter[key], 6)
#         print int(round(10000*cost)), int(10*(time.time() - self.t)), paramafter
#         self.t = time.time()
#         return cost
#
#     def launch(self):
#         result = differential_evolution(self.to_optimise, self.bounds)
#         return result.x, result.fun
