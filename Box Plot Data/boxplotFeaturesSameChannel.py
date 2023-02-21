# Purpose: create box plots for data collected from training on various
#   numbers of features
#	Where the original boxplotFeatures gives one plot, this should yield 5
# Note: to obtain the data for these plots I have performed 25 runs
# 	of 5, 10, 15, and 20 features and manually entered the results in a csv file

import matplotlib.pyplot as plt
import numpy as np
import csv

rows5 = []
rows10 = []
rows15 = []
rows15 = []
rows20 = []

with open('BoxPlot5.csv') as file_name:
	file_read = csv.reader(file_name)

	for row in file_read:
		rows5.append(row)


with open('BoxPlot10.csv') as file_name:
	file_read = csv.reader(file_name)

	for row in file_read:
		rows10.append(row)


with open('BoxPlot15.csv') as file_name:
	file_read = csv.reader(file_name)

	for row in file_read:
		rows15.append(row)


with open('BoxPlot20.csv') as file_name:
	file_read = csv.reader(file_name)

	for row in file_read:
		rows20.append(row)

# print(rows)

# top b
topb5 = np.array(rows5[0])
topb5 = topb5[1:]
topb5 = np.asarray(topb5, dtype=float)

topb10 = np.array(rows10[0])
topb10 = topb10[1:]
topb10 = np.asarray(topb10, dtype=float)

topb15 = np.array(rows15[0])
topb15 = topb15[1:]
topb15 = np.asarray(topb10, dtype=float)

topb20 = np.array(rows20[0])
topb20 = topb20[1:]
topb20 = np.asarray(topb20, dtype=float)

# antitopb
antitopb5 = np.array(rows5[1])
antitopb5 = antitopb5[1:]
antitopb5 = np.asarray(antitopb5, dtype=float)

antitopb10 = np.array(rows10[1])
antitopb10 = antitopb10[1:]
antitopb10 = np.asarray(antitopb10, dtype=float)

antitopb15 = np.array(rows15[1])
antitopb15 = antitopb15[1:]
antitopb15 = np.asarray(antitopb15, dtype=float)

antitopb20 = np.array(rows20[1])
antitopb20 = antitopb20[1:]
antitopb20 = np.asarray(antitopb20, dtype=float)

# top w
topw5 = np.array(rows5[2])
topw5 = topw5[1:]
topw5 = np.asarray(topw5, dtype=float)

topw10 = np.array(rows10[2])
topw10 = topw10[1:]
topw10 = np.asarray(topw10, dtype=float)

topw15 = np.array(rows15[2])
topw15 = topw15[1:]
topw15 = np.asarray(topw15, dtype=float)

topw20 = np.array(rows20[2])
topw20 = topw20[1:]
topw20 = np.asarray(topw20, dtype=float)

# hadronic tau
hadronictau5 = np.array(rows5[3])
hadronictau5 = hadronictau5[1:]
hadronictau5 = np.asarray(hadronictau5, dtype=float)

hadronictau10 = np.array(rows10[3])
hadronictau10 = hadronictau10[1:]
hadronictau10 = np.asarray(hadronictau10, dtype=float)

hadronictau15 = np.array(rows15[3])
hadronictau15 = hadronictau15[1:]
hadronictau15 = np.asarray(hadronictau15, dtype=float)

hadronictau20 = np.array(rows20[3])
hadronictau20 = hadronictau20[1:]
hadronictau20 = np.asarray(hadronictau20, dtype=float)

# antitop lepton
antitoplep5 = np.array(rows5[4])
antitoplep5 = antitoplep5[1:]
antitoplep5 = np.asarray(antitoplep5, dtype=float)

antitoplep10 = np.array(rows10[4])
antitoplep10 = antitoplep10[1:]
antitoplep10 = np.asarray(antitoplep10, dtype=float)

antitoplep15 = np.array(rows15[4])
antitoplep15 = antitoplep15[1:]
antitoplep15 = np.asarray(antitoplep15, dtype=float)

antitoplep20 = np.array(rows20[4])
antitoplep20 = antitoplep20[1:]
antitoplep20 = np.asarray(antitoplep20, dtype=float)


# print("\n")
# print(topb)
# print("\n")
# print(antitopb)
# print("\n")
# print(topw)
# print("\n")
# print(hadronictau)
# print("\n")
# print(antitoplep)

#top b
# collecting data for the box plot
# data = [topb, antitopb, topw, hadronictau, antitoplep]
data = [topb5, topb10, topb15, topb20]
fig = plt.figure(figsize = (10,7))

# ax = fig.add_axes([0, 0, 1, 1])
# bp = ax.boxplot(data)
fig1, ax1 = plt.subplots()
ax1.boxplot(data, vert = True, whis = 0.75)
ax1.set_title('Top b')
plt.xticks([1, 2, 3, 4], ['5', '10', '15', '20'])
plt.xlabel("Top \'x\' Number of Features")
plt.ylabel("Percentage Accuracy")
# ax1.set_ylabel('ylabel', '1', '2', '3', '4')

print("saving top b")
plt.savefig("figures/boxplot_topb.pdf")


#antitop b
# collecting data for the box plot
# data = [topb, antitopb, topw, hadronictau, antitoplep]
data = [antitopb5, antitopb10, antitopb15, antitopb20]
fig = plt.figure(figsize = (10,7))

# ax = fig.add_axes([0, 0, 1, 1])
# bp = ax.boxplot(data)
fig1, ax1 = plt.subplots()
ax1.boxplot(data, vert = True, whis = 0.75)
ax1.set_title('Antitop b')
plt.xticks([1, 2, 3, 4], ['5', '10', '15', '20'])
plt.xlabel("Top \'x\' Number of Features")
plt.ylabel("Percentage Accuracy")
# ax1.set_ylabel('ylabel', '1', '2', '3', '4')

print("saving anti top b")
plt.savefig("figures/boxplot_antitopb.pdf")


#top w
# collecting data for the box plot
# data = [topb, antitopb, topw, hadronictau, antitoplep]
data = [topw5, topw10, topw15, topw20]
fig = plt.figure(figsize = (10,7))

# ax = fig.add_axes([0, 0, 1, 1])
# bp = ax.boxplot(data)
fig1, ax1 = plt.subplots()
ax1.boxplot(data, vert = True, whis = 0.75)
ax1.set_title('Top w')
plt.xticks([1, 2, 3, 4], ['5', '10', '15', '20'])
plt.xlabel("Top \'x\' Number of Features")
plt.ylabel("Percentage Accuracy")
# ax1.set_ylabel('ylabel', '1', '2', '3', '4')

print("saving top w")
plt.savefig("figures/boxplot_topw.pdf")


# hadronic tau
# collecting data for the box plot
# data = [topb, antitopb, topw, hadronictau, antitoplep]
data = [hadronictau5, hadronictau10, hadronictau15, hadronictau20]
fig = plt.figure(figsize = (10,7))

# ax = fig.add_axes([0, 0, 1, 1])
# bp = ax.boxplot(data)
fig1, ax1 = plt.subplots()
ax1.boxplot(data, vert = True, whis = 0.75)
ax1.set_title('Hadronic Tau')
plt.xticks([1, 2, 3, 4], ['5', '10', '15', '20'])
plt.xlabel("Top \'x\' Number of Features")
plt.ylabel("Percentage Accuracy")
# ax1.set_ylabel('ylabel', '1', '2', '3', '4')

print("saving hadronic tau")
plt.savefig("figures/boxplot_hadrnoictau.pdf")


# antitop lepton
# collecting data for the box plot
# data = [topb, antitopb, topw, hadronictau, antitoplep]
data = [antitoplep5, antitoplep10, antitoplep15, antitoplep20]
fig = plt.figure(figsize = (10,7))

# ax = fig.add_axes([0, 0, 1, 1])
# bp = ax.boxplot(data)
fig1, ax1 = plt.subplots()
ax1.boxplot(data, vert = True, whis = 0.75)
ax1.set_title('Antitop Lepton')
plt.xticks([1, 2, 3, 4], ['5', '10', '15', '20'])
plt.xlabel("Top \'x\' Number of Features")
plt.ylabel("Percentage Accuracy")
# ax1.set_ylabel('ylabel', '1', '2', '3', '4')

print("saving antitop lepton")
plt.savefig("figures/boxplot_antitoplepton.pdf")

print("done")
