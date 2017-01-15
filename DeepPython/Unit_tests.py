from unittest import TestCase

import time

from DeepPython import Data, Bookmaker, Launch, Params, Elostd


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("[TEST] {0} took {3} s".format(method.__name__, args, kw, str(int((te - ts) * 1000) * 0.001)[:5]))
        return result

    return timed


class TestSystem(TestCase):
    @timeit
    def test_import_data(self):
        data = Data.Data(Params.FILE)
        self.assertTrue(len(data.py_datas), 7847)
        expected = {'nb_max_journee': 38, 'nb_journee': 912, 'nb_matchs': 7847, 'nb_teams': 42, 'nb_saisons': 23,
                    'min_saison': 1994}
        for n, v in expected.items():
            self.assertTrue(data.meta_datas[n], expected[n])
