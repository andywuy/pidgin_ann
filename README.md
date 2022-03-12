# PIDGIN_with_ANN

## Introduction
This is a re-implementation of the original PIDGIN software (ref: https://github.com/BenderGroup/PIDGINv4) using the Artificial Neural Network instead of Random Forest.
The software does protein target prediction by training the machine model on bioactivity data from ChEMBL (version 26).

## Filetree structure
The `src/` directory contains the source codes.

The `no_ortho/` directory contains the bioactivity dataset for each protein target. 
The datasets are named as `[uniprot_id].smi.zip`.
For each protein target, we have information about a list of compounds.
The compound information includes smile, activity label at four IC50 cut-offs (100μM, 10μM, 1μM and 0.1μM) and so on.
The activity label is 1 if the compound's IC50 value for this protein target is below the threshold, and it is 0 if otherwise.
The datasets are generated without mapping to orthologues. 
Overall, there are 3698 protein targets (datasets).

The `no_ortho/` direcory also contains `classes_in_model_no_ortho.txt`. On each line, we can find the information about each model (e.g. protein_id, model_id, etc.)

The `pidgin_ann_env.yml` is used when setting up the conda virtual environment.

The `Project_report.pdf` is a summary report on the implementation and performance of the artificial neural network.

## Installation
- Download the codes and the datasets to the current directory: `git clone https://github.com/andywuy/PIDGIN_with_ANN.git`.
- Create virtual environment: `conda env create -f pidgin_ann_env.yml --name pidgin_ann_env`. 
- Run `conda activate pidgin_ann_env` to activate the environment. 
- Run `python -m pip install -e .` to install it in development mode.

## Usage
`>>> import pidgin_ann`

To prepare training inputs (e.g. we want to train 5 models):

`>>> m1 = pidgin_ann.Model("no_ortho/bioactivity_datasets", "no_ortho/classes_in_model_no_ortho.txt",
                 "model_inputs", n=5)`

`>>> m1.create()`

To train the models:

`>>> m2 = pidgin_ann.Train("model_inputs", "training_outputs")`

`>>> m2.train()`

## Installation and Usage
- Navigate the directory you wish to install `PIDGIN_with_ANN` in the Linux terminal, then run `git clone https://github.com/andywuy/PIDGIN_with_ANN.git`.
- In the following steps, we work in the root directory `PIDGIN_with_ANN/`.
- Open terminal in Linux and run `conda env create -f pidgin_ann_env.yml --name pidgin_ann_env`. 
- Run `conda activate pidgin_ann_env` to activate the environment. 
- Run `python3 ./scripts/get_model_input.py` to extract data from the bioactivity datasets and generate inputs for the machine learning model training.  The inputs will be saved in `PIDGIN_with_ANN/model_inputs/`. The prompt will ask you to specify the number of model inputs. To train 1000 models, enter 1000.
- Run `python3 ./scripts/ann_training.py`. The prompt will ask you to specify the how many models you want to train. This number must be less or equal to the number of model inputs.
- The outputs will be written to `PIDGIN_with_ANN/training_outputs/`. It includes the trained models and `training_log.txt` which includes performance metrics.

