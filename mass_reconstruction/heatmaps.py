# short script to make some correlation heatmaps of our data
# not necessary for running the mass reco training files

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# # seaborn hearmaps for each part of data,,,
# train_ttH = pd.read_csv("../data/mass_reco/mass_reco_input_narrow_selection_train_ttH.csv")
# train_ttZ = pd.read_csv("../data/mass_reco/mass_reco_input_narrow_selection_train_ttZ.csv")
#
# test_ttH = pd.read_csv("../data/mass_reco/mass_reco_input_narrow_selection_test_ttH.csv")
# test_ttZ = pd.read_csv("../data/mass_reco/mass_reco_input_narrow_selection_test_ttZ.csv")
#
# val_ttH =  pd.read_csv("../data/mass_reco/mass_reco_input_narrow_selection_val_ttH.csv")
# val_ttZ =  pd.read_csv("../data/mass_reco/mass_reco_input_narrow_selection_val_ttZ.csv")
#
# feature_names = pd.read_csv("../data/mass_reco/mass_reco_input_narrow_selection_test_ttH.csv", nrows=1)
# print(feature_names)
# feature_names = list(feature_names.columns)
# print(feature_names)
#
# bad_features = ['top W q2 pz', 'visible main E', 'hadr tau E', 'top W q2 E', 'visible antitop E', 'top W q1 E', 'antitop b E', 'top b E', 'antitop lepton E', 'top E', 'main lepton E', 'true main px', 'true main py', 'top W E', 'antitop b pz']
# temp = set(bad_features)
# index_bad_features = [i for i, val in enumerate(feature_names) if val in temp]
# print(index_bad_features)
#
# train_ttH = train_ttH.drop(bad_features, axis=1)
# train_ttZ = train_ttZ.drop(bad_features, axis=1)
#
# test_ttH = test_ttH.drop(bad_features, axis=1)
# test_ttZ = test_ttZ.drop(bad_features, axis=1)
#
# val_ttH = val_ttH.drop(bad_features, axis=1)
# val_ttZ = val_ttZ.drop(bad_features, axis=1)
#
#
# all_data = pd.concat([train_ttH, train_ttZ, test_ttH, test_ttZ, val_ttH, val_ttZ])
#
# fig, ax = plt.subplots(figsize=(100,100))
#
# corr_plot = sns.heatmap(train_ttH.corr(), annot=True, square=True)
# plt.savefig("figures/corr_plot_ttH_ttZ_feature_removed.pdf")
# plt.savefig("figures/corr_plot_ttH_ttZ_feature_removed.png")


# average of 5 runs for regular and features removed
# load everything from the regular runs and take the average
# background
regular_bg = np.load('samples/correct_bg_regular0.npy')
for x in range(4):
	name = 'samples/correct_bg_regular' + str((x+1)) + ".npy"
	regular_bg += np.load(name)

regular_bg = regular_bg / 5

# Higgs
regular_h = np.load('samples/correct_h_regular0.npy')
for x in range(4):
	name = 'samples/correct_h_regular' + str((x+1)) + ".npy"
	regular_h += np.load(name)

regular_h = regular_h / 5

# weighted
regular_weighted = np.load('samples/correct_weighted_regular0.npy')
for x in range(4):
	name = 'samples/correct_weighted_regular' + str((x+1)) + ".npy"
	regular_weighted += np.load(name)

regular_weighted = regular_weighted / 5

regular_steps = np.load('samples/correct_steps_regular0.npy')
for x in range(4):
	name = 'samples/correct_steps_regular' + str((x+1)) + ".npy"
	regular_steps += np.load(name)

regular_steps = regular_steps / 5





# load everything from the runs with the feature importance removed and take the average
featureImport_bg = np.load('samples/correct_bg_featureImport0.npy')
for x in range(4):
	name = 'samples/correct_bg_featureImport' + str((x+1)) + ".npy"
	featureImport_bg += np.load(name)

featureImport_bg = featureImport_bg / 5

# Higgs
featureImport_h = np.load('samples/correct_h_featureImport0.npy')
for x in range(4):
	name = 'samples/correct_h_featureImport' + str((x+1)) + ".npy"
	featureImport_h += np.load(name)

featureImport_h = featureImport_h / 5

# weighted
featureImport_weighted = np.load('samples/correct_weighted_featureImport0.npy')
for x in range(4):
	name = 'samples/correct_weighted_regular' + str((x+1)) + ".npy"
	featureImport_weighted += np.load(name)

featureImport_weighted = featureImport_weighted / 5

featureImport_steps = np.load('samples/correct_steps_featureImport0.npy')
for x in range(4):
	name = 'samples/correct_steps_regular' + str((x+1)) + ".npy"
	featureImport_steps += np.load(name)

featureImport_steps = featureImport_steps / 5


# create the figure


# regular
correct_h_area = np.trapz(regular_h, regular_steps)
correct_bg_area = np.trapz(regular_bg, regular_steps)
correct_weighted_area = np.trapz(regular_weighted, regular_steps)


legend_h = "correct Higgs (area: " + str(round(correct_h_area, 2)) + " )"
legend_bg = "correct background (area: " + str(round(correct_bg_area, 2)) + " )"
legend_weighted = "weighted correct total (area: " + str(round(correct_weighted_area, 2)) + " )"


plt.figure(figsize=(12,8))
plt.plot(regular_steps, regular_h, linewidth=3, color='blue')
plt.plot(regular_steps, regular_bg, linewidth=3, color='orange')
plt.plot(regular_steps, regular_weighted, linewidth=3, color='magenta')
plt.title("Mass reconstruction separation curve - NN")
plt.xlabel("separation mass [GeV]")
plt.ylabel("percentage correct [%]")
plt.legend([legend_h, legend_bg, legend_weighted])
plt.grid()
plt.savefig("../figures/mass_histo_narrow_separation_btags_regular5.pdf")
plt.show()


# feature importance
correct_h_area = np.trapz(featureImport_h, featureImport_steps)
correct_bg_area = np.trapz(featureImport_bg, featureImport_steps)
correct_weighted_area = np.trapz(featureImport_weighted, featureImport_steps)


legend_h = "correct Higgs (area: " + str(round(correct_h_area, 2)) + " )"
legend_bg = "correct background (area: " + str(round(correct_bg_area, 2)) + " )"
legend_weighted = "weighted correct total (area: " + str(round(correct_weighted_area, 2)) + " )"


plt.figure(figsize=(12,8))
plt.plot(featureImport_steps, featureImport_h, linewidth=3, color='blue')
plt.plot(featureImport_steps, featureImport_bg, linewidth=3, color='orange')
plt.plot(featureImport_steps, featureImport_weighted, linewidth=3, color='magenta')
plt.title("Mass reconstruction separation curve - NN")
plt.xlabel("separation mass [GeV]")
plt.ylabel("percentage correct [%]")
plt.legend([legend_h, legend_bg, legend_weighted])
plt.grid()
plt.savefig("../figures/mass_histo_narrow_separation_btags_featureImport5.pdf")
plt.show()
