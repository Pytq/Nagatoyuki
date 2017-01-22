import csv
import matplotlib.pyplot as plt
import math
import numpy as np

class Stats:

    def __init__(self, filename, s=1.):
        self.filename = filename
        y = []
        with open(filename) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
               y.append(100*float(row[0]))
        wind = np.array([math.exp(-((i-20.)/s)**2)/(s*math.sqrt(2*math.pi)) for i in range(40)])
        y_conv = np.convolve(np.array(y), wind)[40:-40]

        x = np.array(range(len(y)))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]

        plt.plot(y_conv,'o', label='Original data')
        plt.plot(x, m * x + c, 'r', label='Fitted line')

        print(40*m,c)
        print(len(x))
        print(len(wind))
        plt.show()

