Source codes and materials for the thesis Application of Machine Learning for the Higgs Boson Mass Reconstruction Using ATLAS Data
For instructions on how to run this code, please refer to the file ttHMassRecoDocs.docx

./data is a folder that will contain all csv and numpy files used by the NNs

./figures is a folder for all pdf files generated by scripts

./mass_reconstruction is a folder for scripts related to the mass reco NNs

./MMC contains MMC script and evaluation

./particle_assignment is a folder for scripts related to the assignment NN training and data extraction

./root_data_extraction contains scripts for extracting csv and numpy files used by NNs from ROOT ntuples

./scaler_params contains parameters of scalers used by NNs

./tool contains other help files

./trained_NN_models will contain trained models

Adam Herold 2021
Emily Hannon 2022 (minor code additions)

NOTE: There are several filetypes excluded in the project .gitignore. 
	These should be excluding filetypes containg large amounts of data and no files essential to the project function.
