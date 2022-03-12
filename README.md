# PIDGIN_with_ANN

## Introduction
This is a re-implementation of the original PIDGIN software (ref: https://github.com/BenderGroup/PIDGINv4) using the Artificial Neural Network instead of Random Forest.
The software does protein target prediction by training the machine model on bioactivity data from ChEMBL (version 26).

## Filetree structure
The `scripts/` directory contains the main programs.

The `mids_list.txt` contains the list of IDs of the 1000 datasets used in the machine learning model training. The ID is named in the form `"[uniprot_id],[SE]_[IC50]"`, where `[uniprot_id]` is the id of the protein in the UniProt database, `[SE]` indicates whether spherical exclusion is used, `[IC50]` is the IC50 value in micromole.

The `no_ortho/` directory contains the bioactivity data. For each protein target, we generate bioactivity datasets at four different IC50 cut-offs: 100μM, 10μM, 1μM and 0.1μM. Each dataset contains a list of compounds which are marked as either active or inactive for the protein target at the chosen IC50 cutoff. The datasets are generated without mapping to orthologues. Overall, there are 3698 protein targets and 11782 bioactivity datasets.

The `no_ortho/` direcory also contains `classes_in_model_no_ortho.txt`. On each line, we can find the information about each model (e.g. protein_id, model_id, etc.)

The `pidgin_ann_env.yml` is used when setting up the conda virtual environment.

The `Project_report.pdf` is a summary report on the implementation and performance of the artificial neural network.

## Installation and Usage
- Navigate the directory you wish to install `PIDGIN_with_ANN` in the Linux terminal, then run `git clone https://github.com/andywuy/PIDGIN_with_ANN.git`.
- In the following steps, we work in the root directory `PIDGIN_with_ANN/`.
- Open terminal in Linux and run `conda env create -f pidgin_ann_env.yml --name pidgin_ann_env`. 
- Run `conda activate pidgin_ann_env` to activate the environment. 
- Run `python3 ./scripts/get_model_input.py` to extract data from the bioactivity datasets and generate inputs for the machine learning model training.  The inputs will be saved in `PIDGIN_with_ANN/model_inputs/`. The prompt will ask you to specify the number of model inputs. To train 1000 models, enter 1000.
- Run `python3 ./scripts/ann_training.py`. The prompt will ask you to specify the how many models you want to train. This number must be less or equal to the number of model inputs.
- The outputs will be written to `PIDGIN_with_ANN/training_outputs/`. It includes the trained models and `training_log.txt` which includes performance metrics.

