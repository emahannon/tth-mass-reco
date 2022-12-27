from ROOT import TFile, TNtuple
import csv
import numpy as np

""" This file has been used to extract real world (non simulation) data from
	ROOT files. """

# f = TFile.Open("../../../../CERN_data/truth/nominal/mc16a/p4416/410155.root")   #or 16,17,18

# This file path must be modified to point to your data that you wish to extract.
f = TFile.Open("../../../../CERN_data/ttH/user.nguseyno.28484942._000001.output.root")   #or 16,17,18


if f.IsZombie() or not f.IsOpen():
	print("Error opening file")

# Note: These variable names may have to be modified if we use a different
# 	version of data and the variable names have been changed.
# Reduced variables without the jets information.
used_ntuple_variables = ["nJets_OR_TauOR", "nJets_OR_DL1r_70", "l2SS1tau", "met_met", "met_phi", \
							"lep_Pt_0", "lep_E_0", "lep_Eta_0", "lep_Phi_0", \
							"lep_Pt_1", "lep_E_1", "lep_Eta_1", "lep_Phi_1", \
							"taus_pt_0", "taus_eta_0", "taus_phi_0", \
							"jet_pt0", "jet_eta0", \
							"jet_pt1", "jet_eta1", \
							"jet_pt2", "jet_eta2", \
							"bjet_pt0", "bjet_eta0", \
							"m_truth_m", "m_truth_pt", "m_truth_eta", "m_truth_phi", "m_truth_e", \
							"m_truth_pdgId", "m_truth_status", "m_truth_barcode", \
							"HT", "taus_numTrack_0", "eventNumber","lep_ID_0","lep_isMedium_0","lep_isolationFCLoose_0","passPLIVVeryTight_0", \
							"lep_isTightLH_0","lep_chargeIDBDTResult_recalc_rel207_tight_0","passPLIVVeryTight_0","lep_ID_1","lep_isMedium_1","lep_isolationFCLoose_1","passPLIVVeryTight_1","lep_isTightLH_1", \
							"lep_RadiusCO_1","lep_Mtrktrk_atPV_CO_1","lep_ambiguityType_1","lep_Mtrktrk_atConvV_CO_0","lep_RadiusCO_0","lep_Mtrktrk_atPV_CO_0","lep_ambiguityType_0","lep_chargeIDBDTResult_recalc_rel207_tight_1", \
							"dilep_type","nTaus_OR_Pt25","lep_Mtrktrk_atConvV_CO_1", \
						]

tree = f.nominal
tree.SetBranchStatus("*",0)

for var_name in used_ntuple_variables:
		tree.SetBranchStatus(var_name,1)

c=0
rows = []
for event in tree:
	# The strict selection similar to the decay channel tag & num of jets requirement.
	if ((abs(event.lep_ID_0)==13 and ord(event.lep_isMedium_0) and ord(event.lep_isolationFCLoose_0) and event.passPLIVVeryTight_0) or (abs(event.lep_ID_0)==11 and ord(event.lep_isolationFCLoose_0) and ord(event.lep_isTightLH_0) and event.lep_chargeIDBDTResult_recalc_rel207_tight_0>0.7 and event.passPLIVVeryTight_0)) and ((abs(event.lep_ID_1)==13 and ord(event.lep_isMedium_1) and ord(event.lep_isolationFCLoose_1) and event.passPLIVVeryTight_1) or (abs(event.lep_ID_1)==11 and ord(event.lep_isolationFCLoose_1) and ord(event.lep_isTightLH_1) and event.lep_chargeIDBDTResult_recalc_rel207_tight_1>0.7 and event.passPLIVVeryTight_1)) and (((abs(event.lep_ID_0) == 13) or ( abs( event.lep_ID_0 ) == 11 and ord(event.lep_ambiguityType_0) == 0 and ( not ((event.lep_Mtrktrk_atPV_CO_0<0.1 and event.lep_Mtrktrk_atPV_CO_0>0) and  not (event.lep_RadiusCO_0>20 and (event.lep_Mtrktrk_atConvV_CO_0<0.1 and event.lep_Mtrktrk_atConvV_CO_0>0))) and  not (event.lep_RadiusCO_0>20 and (event.lep_Mtrktrk_atConvV_CO_0<0.1 and event.lep_Mtrktrk_atConvV_CO_0>0))))) and ((abs( event.lep_ID_1 ) == 11 and ord(event.lep_ambiguityType_1) == 0 and  not ((event.lep_Mtrktrk_atPV_CO_1<0.1 and event.lep_Mtrktrk_atPV_CO_1>0) and  not (event.lep_RadiusCO_1>20 and (event.lep_Mtrktrk_atConvV_CO_1<0.1 and event.lep_Mtrktrk_atConvV_CO_1>0))) and  not (event.lep_RadiusCO_1>20 and (event.lep_Mtrktrk_atConvV_CO_1<0.1 and event.lep_Mtrktrk_atConvV_CO_1>0))) or (abs(event.lep_ID_1) == 13))) and ord(event.nTaus_OR_Pt25)>=1 and (ord(event.nJets_OR_TauOR)>2 and ord(event.nJets_OR_DL1r_70)>0) and (event.dilep_type and event.lep_ID_0*event.lep_ID_1>0):
		c +=1
		row = []
		row.append(float(event.met_met))
		row.append(float(event.met_phi))
		row.append(float(event.lep_Pt_0))
		row.append(float(event.lep_E_0))
		row.append(float(event.lep_Eta_0))
		row.append(float(event.lep_Phi_0))
		row.append(float(event.lep_Pt_1))
		row.append(float(event.lep_E_1))
		row.append(float(event.lep_Eta_1))
		row.append(float(event.lep_Phi_1))
		row.append(float(event.taus_pt_0))
		row.append(float(event.taus_eta_0))
		row.append(float(event.taus_phi_0))
		row.append(float(event.jet_pt0))
		row.append(float(event.jet_eta0))
		row.append(float(event.jet_pt1))
		row.append(float(event.jet_eta1))
		row.append(float(event.jet_pt2))
		row.append(float(event.jet_eta2))
		row.append(float(event.bjet_pt0))
		row.append(float(event.bjet_eta0))
		row.append(float(event.HT))
		row.append(float(ord(event.taus_numTrack_0)))
		row.append(float(event.eventNumber))
		rows.append(row)

f = open("user.nguseyno.28484942._000001.output.csv", "a")   # 16,17,18
writer = csv.writer(f)
writer.writerows(rows)
f.close()
# print(c)
