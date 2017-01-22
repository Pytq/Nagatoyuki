import csv
import matplotlib.pyplot as plt



class Stats:

    def __init__(self, filename):
        self.filename = filename
        x = []
        with open(filename) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
               x.append(row[1])
        print(x)
        plt.plot(x)
        plt.show()
        self.x = x

