import csv
from DeepPython import ToolBox, Params, Slice

ALL_FEATURES = ['saison', 'team_h', 'team_a', 'res', 'score_h', 'score_a',
                'odd_win_h', 'odd_tie', 'odd_los_h', 'odds', 'home_matchid', 'away_matchid']


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

    def next_slice(self, group):
        self.__slices[group].next_slice()

    def get_slice(self, group, feed_dict=None):
        fun = self.__slices[group].get_slice
        return self.__get_slice(fun, group, feed_dict=feed_dict)

    def cget_slice(self, group, feed_dict=None):
        fun = self.__slices[group].cget_slice
        return self.__get_slice(fun, group, feed_dict=feed_dict)

    def get_both_slices(self, group, train_p=None, test_p=None):
        fun = self.get_slice
        return self.__get_both_slices(fun, group, train_p=train_p, test_p=test_p)

    def cget_both_slices(self, group, train_p=None, test_p=None):
        fun = self.cget_slice
        return self.__get_both_slices(fun, group, train_p=train_p, test_p=test_p)

    def __get_slice(self, fun, group, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}
        if group not in self.__slices:
            self.init_slices(group)
        extract_slice = fun(feed_dict)
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

    def __get_both_slices(self, fun, group, train_p=None, test_p=None):
        if train_p is None:
            train_p = {}
        if test_p is None:
            test_p = {}
        train_p['label'] = 'train'
        test_p['label'] = 'test'
        # self.__slices[group].check_slices(train_p, test_p)
        train = fun(group, train_p)
        test = fun(group, test_p)
        return train, test

    def nb_slices(self, group):
        return self.__slices[group].nb_slices

    # def shuffle_slice(self, group):
    #     if group not in self.__slices:
    #         self.init_slices(group)
    #     self.__slices[group].shuffle_slice()

    def __get_datas(self, filename):
        self.meta_datas["nb_matchs"] = 0
        max_teamid = 0
        min_teamid = float("inf")
        min_saison = float("inf")
        max_saison = 0
        max_journee = 0
        self.team_to_id = {}
        self.id_to_team = []
        team_nb_matchs = {}
        teams_this_saison = set([])
        teams_per_saison = []
        with open(filename) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            header = spamreader.__next__()
            for row in spamreader:
                if row:
                    dict_row = self.__add_header_to_data(row, header)
                    if dict_row['HomeTeam'] not in self.team_to_id:
                        self.team_to_id[dict_row['HomeTeam']] = len(self.id_to_team)
                        self.id_to_team.append(dict_row['HomeTeam'])
                        team_nb_matchs[dict_row['HomeTeam']] = 0
                    if dict_row['AwayTeam'] not in self.team_to_id:
                        self.team_to_id[dict_row['AwayTeam']] = len(self.id_to_team)
                        self.id_to_team.append(dict_row['AwayTeam'])
                        team_nb_matchs[dict_row['AwayTeam']] = 0
                    dict_row['HomeId'] = team_nb_matchs[dict_row['HomeTeam']]
                    dict_row['AwayId'] = team_nb_matchs[dict_row['AwayTeam']]
                    team_nb_matchs[dict_row['HomeTeam']] += 1
                    team_nb_matchs[dict_row['AwayTeam']] += 1
                    if dict_row['saison'] < max_saison:
                        raise Exception('Saisons non croissantes')
                    if dict_row['saison'] > max_saison:
                        max_saison = dict_row['saison']
                        teams_per_saison.append(list(teams_this_saison))
                        teams_this_saison = set([])
                    teams_this_saison.add(dict_row['AwayTeam'])
                    teams_this_saison.add(dict_row['HomeTeam'])
                    if dict_row['saison'] < min_saison:
                        min_saison = dict_row['saison']
                    self.py_datas.append(dict_row)
                    self.meta_datas["nb_matchs"] += 1
        self.meta_datas["nb_teams"] = len(self.id_to_team)
        self.meta_datas["min_saison"] = min_saison
        self.meta_datas["nb_saisons"] = 1 + max_saison - min_saison
        self.meta_datas["nb_max_journee"] = max_journee
        self.meta_datas["nb_journee"] = (self.meta_datas["nb_saisons"] + 1) * self.meta_datas["nb_max_journee"]
        print(team_nb_matchs)
        self.meta_datas["max_match_id"] = max(list(team_nb_matchs.values()))

        print("Loading data with {}".format(self.meta_datas))
        print('nb team per season: ', [len(x) for x in teams_per_saison])

    def __get_formated_datas(self):
        for dict_row in self.py_datas:
            self.__datas['odd_win_h'].append([dict_row["BbMxH"]])
            self.__datas['odd_tie'].append([dict_row["BbMxD"]])
            self.__datas['odd_los_h'].append([dict_row["BbMxA"]])
            self.__datas['odds'].append([dict_row["BbMxH"], dict_row["BbMxD"], dict_row["BbMxA"]])
            self.__datas['saison'].append(ToolBox.make_vector(dict_row["saison"]-self.meta_datas['min_saison'], self.meta_datas['nb_saisons']))
            self.__datas['team_h'].append(ToolBox.make_vector(self.team_to_id[dict_row["HomeTeam"]], self.meta_datas["nb_teams"]))
            self.__datas['team_a'].append(ToolBox.make_vector(self.team_to_id[dict_row["AwayTeam"]], self.meta_datas["nb_teams"]))
            self.__datas['home_matchid'].append(ToolBox.make_vector(dict_row["HomeId"], self.meta_datas["max_match_id"]))
            self.__datas['away_matchid'].append(ToolBox.make_vector(dict_row["AwayId"], self.meta_datas["max_match_id"]))
            score_team_h = min(int(dict_row["scoreH"]), Params.MAX_GOALS)
            score_team_a = min(int(dict_row["scoreA"]), Params.MAX_GOALS)
            self.__datas['score_h'].append(ToolBox.make_vector(score_team_h, Params.MAX_GOALS+1))
            self.__datas['score_a'].append(ToolBox.make_vector(score_team_a, Params.MAX_GOALS+1))
            self.__datas['res'].append(ToolBox.result_vect(int(dict_row["scoreH"]) - int(dict_row["scoreA"])))

    def __add_header_to_data(self, row, header):
        d = {}
        if len(row) < len(header):
            raise Exception('Too many header', row, header, len(row), len(header))
        for i in range(len(header)):
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
        dict_row["saison"] = int(dict_row["Season"])
        dict_row["scoreH"] = int(dict_row["FTHG"])
        dict_row["scoreA"] = int(dict_row["FTAG"])

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



