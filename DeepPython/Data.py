import csv
import ToolBox
import Slice
import Params

ALL_FEATURES = ['saison', 'team_h', 'team_a', 'res', 'score_h', 'score_a', 'journee',
                'odd_win_h', 'odd_tie', 'odd_los_h', 'odds']


class Data:

    def __init__(self, filename):
        self.__datas = {}
        self.__slices = {}
        self.meta_datas = {}
        self.py_datas = []

        for key in ALL_FEATURES:
            self.__datas[key] = []

        self.__get_datas(filename)
        self.__get_formated_datas()
        Data.check_len(self.__datas)

    def init_slices(self, group, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}
        self.__slices[group] = Slice.Slice(self.py_datas, group, feed_dict=feed_dict)

    def get_slice(self, group, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}
        if group not in self.__slices:
            self.init_slices(group)
        extract_slice = self.__slices[group].get_slice(feed_dict)
        s = {}
        for key in self.__datas:
            s[key] = extract_slice(self.__datas[key])
        Data.check_len(s)
        if 'when_odd' in feed_dict and feed_dict['when_odd']:
            s2 = {}
            for key in s:
                s2[key] = []
            for i in range(len(s['odd_tie'])):
                if s['odd_tie'][i][0] >= 0.:
                    for key in s2:
                        s2[key].append(s[key][i])
            s = s2
        return s

    def nb_slices(self, group):
        return self.__slices[group].nb_slices

    def shuffle_slice(self, group):
        self.__slices[group].shuffle_slice()

    def __get_datas(self, filename):
        self.meta_datas["nb_matchs"] = 0
        max_teamid = 0
        min_teamid = float("inf")
        min_saison = float("inf")
        max_saison = 0
        max_journee = 0
        with open(filename) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            header = spamreader.__next__()
            for row in spamreader:
                if row:
                    dict_row = self.__add_header_to_data(row, header)
                    if dict_row['id1'] > max_teamid or dict_row['id2'] > max_teamid:
                        max_teamid = max(dict_row['id1'], dict_row['id2'])
                    if dict_row['id1'] < min_teamid or dict_row['id2'] < min_teamid:
                        min_teamid = min(dict_row['id1'], dict_row['id2'])
                    if dict_row['saison'] > max_saison:
                        max_saison = dict_row['saison']
                    if dict_row['journee'] > max_journee:
                        max_journee = dict_row['journee']
                    if dict_row['saison'] < min_saison:
                        min_saison = dict_row['saison']
                    self.py_datas.append(dict_row)
                    self.meta_datas["nb_matchs"] += 1
        if min_teamid != 0:
            raise 'id error'
        self.meta_datas["nb_teams"] = max_teamid + 1
        self.meta_datas["min_saison"] = min_saison
        self.meta_datas["nb_saisons"] = 1 + max_saison - min_saison
        self.meta_datas["nb_max_journee"] = max_journee
        self.meta_datas["nb_journee"] = (self.meta_datas["nb_saisons"] + 1) * self.meta_datas["nb_max_journee"]
        print(self.meta_datas)

    def __get_formated_datas(self):
        for dict_row in self.py_datas:
            self.__datas['odd_win_h'].append([dict_row["BbMxH"]])
            self.__datas['odd_tie'].append([dict_row["BbMxD"]])
            self.__datas['odd_los_h'].append([dict_row["BbMxA"]])
            self.__datas['odds'].append([dict_row["BbMxH"], dict_row["BbMxD"], dict_row["BbMxA"]])
            self.__datas['saison'].append(ToolBox.make_vector(dict_row["saison"]-self.meta_datas['min_saison'], self.meta_datas['nb_saisons']))
            self.__datas['team_h'].append(ToolBox.make_vector(dict_row["id1"], self.meta_datas["nb_teams"]))
            self.__datas['team_a'].append(ToolBox.make_vector(dict_row["id1"], self.meta_datas["nb_teams"]))
            score_team_h = min(int(dict_row["score1"]), Params.MAX_GOALS)
            score_team_a = min(int(dict_row["score2"]), Params.MAX_GOALS)
            self.__datas['score_h'].append(ToolBox.make_vector(score_team_h, Params.MAX_GOALS+1))
            self.__datas['score_a'].append(ToolBox.make_vector(score_team_a, Params.MAX_GOALS+1))
            self.__datas['res'].append(ToolBox.result_vect(int(dict_row["score1"]) - int(dict_row["score2"])))
            journee = dict_row["journee"] + self.meta_datas["nb_max_journee"]*(dict_row["saison"]-self.meta_datas['min_saison'])
            self.__datas['journee'].append(ToolBox.make_vector(journee, self.meta_datas['nb_journee']))

    def __add_header_to_data(self, row, header):
        d = {}
        if len(row) != len(header):
            raise Exception('Data.py associate_data_to_header()')
        for i in range(len(row)):
            d[header[i].replace(" ", "")] = row[i]
        self.__format_row(d)
        return d

    def __format_row(self, dict_row):
        if dict_row["BbMxH"] == '':
            dict_row["BbMxH"] = -1.
            dict_row["BbMxD"] = -1.
            dict_row["BbMxA"] = -1.
        else:
            dict_row["BbMxH"] = float(dict_row["BbMxH"])
            dict_row["BbMxD"] = float(dict_row["BbMxD"])
            dict_row["BbMxA"] = float(dict_row["BbMxA"])
        dict_row["saison"] = int(dict_row["saison"])
        dict_row["journee"] = int(dict_row["journee"])
        dict_row["id1"] = int(dict_row["id1"]) - 1
        dict_row["id2"] = int(dict_row["id2"]) - 1
        dict_row["score1"] = int(dict_row["score1"])
        dict_row["score2"] = int(dict_row["score2"])

    @staticmethod
    def check_len(s):
        check_len = []
        for key in s:
            check_len.append(len(s[key]))
        if check_len[1:] != check_len[:-1]:
            raise Exception('Data.py check_len')

    @staticmethod
    def is_empty(s):
        return s[list(s.keys())[0]] == []



