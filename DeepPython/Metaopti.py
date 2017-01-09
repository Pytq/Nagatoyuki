import random
from copy import copy


class MetaOpti:
    def __init__(self, fun, params, reset, k=5, alpha=0.8, treshold=0.001):
        self.fun = fun
        self.params = {}
        self.paramrange = {}
        for key in params:
            self.params[key] = 0.
            self.paramrange[key] = 0.5
        self.reset = reset
        self.k = k
        self.alpha = alpha
        self.treshold = treshold

    def init_paramrange(self):
        self.reset()
        x = copy(self.params)
        for key in self.params:
            prange = 0
            for sign in [-1., 1.]:
                x[key] = 0.
                i = float("inf")
                j = float("inf")
                z = self.fun(x)
                while (z <= j or j <= i) and abs(x[key]) < 10**8:
                    i = j
                    j = z
                    x[key] = x[key] * 2. + sign
                    z = self.fun(x)
                if x[key] >= 10**8:
                    raise 'out of range'
                prange = max(prange, abs(x[key]))
            self.paramrange[key] = prange
            x[key] = 0.

    def map_paramrange(self, mapper):
        for key in self.paramrange:
            self.paramrange[key] = mapper(self.paramrange[key])

    def opti_step(self):
        x = copy(self.params)
        keys = random.shuffle(self.params.keys())
        self.reset()
        start_value = self.fun(self.params)
        while keys:
            key = keys.pop()
            y_min = float("inf")
            x_min = x[key]
            y_start = self.fun(self.params)
            for k in range(self.k):
                x[key] = self.params[key] + self.paramrange[key] * (float(k)/(self.k-1.) - 0.5)
                y = self.fun(x)
                if y < y_min:
                    y_min = y
                    x_min = x[key]
            self.paramrange[key] *= self.alpha
            self.params[key] = x_min
            if y_start - y_min < self.treshold:
                del self.params[key]
        current_value = self.fun(self.params)
        return start_value - current_value, current_value


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
