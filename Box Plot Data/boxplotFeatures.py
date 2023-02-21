# Purpose: create box plots for data collected from training on various
#   numbers of features
# Note: to obtain the data for these plots I have performed 25 runs
# 	of 5, 10, 15, and 20 features and manually entered the results in a csv file

import matplotlib.pyplot as plt
import numpy as np
import csv

rows = []

with open('BoxPlot20.csv') as file_name:
	file_read = csv.reader(file_name)

	for row in file_read:
		rows.append(row)

print(rows)

topb = np.array(rows[0])
topb = topb[1:]
topb = np.asarray(topb, dtype=float)

antitopb = np.array(rows[1])
antitopb = antitopb[1:]
antitopb = np.asarray(antitopb, dtype=float)

topw = np.array(rows[2])
topw = topw[1:]
topw = np.asarray(topw, dtype=float)

hadronictau = np.array(rows[3])
hadronictau = hadronictau[1:]
hadronictau = np.asarray(hadronictau, dtype=float)

antitoplep = np.array(rows[4])
antitoplep = antitoplep[1:]
antitoplep = np.asarray(antitoplep, dtype=float)

print("\n")
print(topb)
print("\n")
print(antitopb)
print("\n")
print(topw)
print("\n")
print(hadronictau)
print("\n")
print(antitoplep)


# collecting data for the box plot
data = [topb, antitopb, topw, hadronictau, antitoplep]
fig = plt.figure(figsize = (10,7))

# ax = fig.add_axes([0, 0, 1, 1])
# bp = ax.boxplot(data)
fig1, ax1 = plt.subplots()
ax1.boxplot(data, vert = True, whis = 0.75)
ax1.set_title('15 Most Significant Features')
plt.xticks([1, 2, 3, 4, 5], ['top b', 'antitop b', 'top w', 'hadronic tau', 'antitop lepton'])
# ax1.set_ylabel('ylabel', '1', '2', '3', '4')

print("saving boxplot15")
plt.savefig("figures/boxplot20.pdf")
