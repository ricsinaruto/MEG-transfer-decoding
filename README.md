# MEG-transfer-decoding
General PhD research code for modeling and decoding M/EEG data. This readme describes how to reproduce results in [Interpretable full-epoch multiclass decoding for M/EEG](https://arxiv.org/abs/2205.14102). We propose that a full-epoch multiclass model is better than sliding window and/or pairwise models for decoding visual stimuli, and we show how it can be used for multivariate pattern analysis (MVPA) through permutation feature importance (PFI).

**Since this repository is constantly evolving and contains a lot of other projects, make sure you are using the code version ([v0.1-paper](https://github.com/ricsinaruto/MEG-transfer-decoding/tree/v0.1-paper)) specifically created for the publication.**

![Animation](https://github.com/ricsinaruto/MEG-transfer-decoding/blob/main/notebooks/mvpa_figures/animation.gif)

## Features
  :magnet: &nbsp; Train linear classifiers on MEG data.  
  :rocket: &nbsp; Can easily specify full-epoch/sliding-window, multiclass/pairwise mode, and extend to other classifiers.  
  :brain: &nbsp; Neuroscientifically interpretable features can be extracted and visualized from trained models.  
  :twisted_rightwards_arrows: &nbsp; Flexible pipeline with support for [multi-run modes](https://github.com/ricsinaruto/MEG-group-decode/edit/main/README.md#multi-run-modes) (e.g. cross-validation within and across subjects, different training set ratios, etc.)
  
## Table of Contents
- [Usage](#usage)
- [Multi-run modes](#multi-run-modes)
- [Data](#data)
- [Models](#models)
- [Arguments](#arguments)
- [Examples](#examples)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)


## Usage
First requirements need to be installed.
```
pip install -r requirements.txt
```

For each run modify ```args.py``` to specify parameters and behaviour, then run ```launch.py``` which calls ```training.py```, which contains the main experimental pipeline:
```
python launch.py
```
Set the function you want to run to True in the ```func``` dictionary in ```args.py```.
Some relevant functions that are available:
* **train**: Trains a specified classifier on the specified dataset. This should only be used with a neural network model.
* **LDA_baseline**: Trains any non-neural network model (LDA, LogisticRegression, etc.) on the specified dataset for multiclass classification.
* **LDA_pairwise**: Same as LDA_baseline, but the models are trained for pairwise classification across all pairs of classes.
* **LDA_channel**: Same as LDA_baseline, but individual models are trained on individual channels of the MEG data, i.e. sliding-channel decoding.
* **PFIts**: runs temporal PFI for a trained model.
* **PFIch**: runs spatial or spatiotemporal PFI for a trained model, depending on input arguments
* **PFIfreq**: runs spectral PFI for a trained model.
* **PFIfreq_ch**: runs spatiospectral PFI for a trained model.
* **PFIfreq_ts**: runs temporospectral PFI for a trained model.
* **multi2pair**: Generates pairwise accuracies across all pairs of classes for a trained multiclass model (on the validation data).

Generally a model and dataset (specified in the corresponding variables in ```args.py```) is required to run anything. The dataset class used for this paper is always ```CichyData```.

## Multi-run modes
To facilitate running multiple trainings with the same command, the following variables in ```args.py``` can be lists:  
```load_model```, ```result_dir```, ```data_path```, ```dump_data```, ```load_data```, ```subjects_data```  
This allows for e.g. running the same model on multiple subjects with a single call.

To run on a dataset with multiple training set ratios, the following parameters can also be lists:  
```max_trials```, ```learning_rate```, ```batch_size```

To run cross-validation, set ```split``` to a list of validation split ratios.

## Data
To preprocess continuous .fif data use ```scripts/cichy_preprocess.py```. This filters the data and creates trials.  
To download and preprocess the publicly available 118-class epoched data run ```scripts/cichy_download.py``` and ```scripts/cichy_preproc_epoched.py```. To download and preprocess the 92-class dataset run ```scripts/cichy92_download.py``` and ```scripts/cichy92_preproc_epoched.py```

If running any function on some data for the first time ```load_data``` needs to be False. This loads preprocessed data given by ```data_path``` variable, and saves ready-to-use splits to ```dump_data``` path. This ready-to-use data can then be used in subsequent runs by setting ```load_data``` to the ```dump_data``` path. Note that when using ```load_data```, the ```num_channels``` parameter should be increased by 1, to account for the target variable.

## Models
Models are run on individual subjects. Please see our [group-level decoding](https://github.com/ricsinaruto/MEG-group-decode) work for running models on groups of subjects.
The following classification models are available in *classifiers_simpleNN.py*:
* ```SimpleClassifier```: Implements an n-layer fully-connected neural network (with an arbitrary activation function).

The following classification models are available in *classifiers_linear.py*:
* ```LDA```: Implements the Linear Discriminant Analysis model.
* ```LogisticReg```: Implements the Logistic Regression model.
* ```SVM```: Implements the Support Vector Machine model.


## Arguments

## Examples
To replicate some of the results in the paper we provide args files in the examples folder. To try these out on the publicly available MEG data, follow these steps:  
1. ```python scripts/cichy_download.py``` to download data.
2. ```python scripts/cichy_preproc_epoched.py``` to preprocess data.
3. Copy the contents of the example args file you want to run into ```args.py```
4. ```python launch.py```

The following example args files are available:
* ```args_linear_subject.py```: trains 

Steps 1 and 2 can be skipped if running on non-public data. The relevant data paths in the args file have to modified in this case. Note that results from running the examples will not 100% reproduce our results, because we used the raw continuous MEG data. Also, different random seeds may cause (very) small differences.

Please note that the variable "n" in the args files controls the number of subjects. This is set to 1 by default. If you want to run on all 15 subjects just change this to 15.

## Visualizations


## Contributing


## Authors
* **[Richard Csaky](https://ricsinaruto.github.io)**

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ricsinaruto/MEG-group-decode/blob/master/LICENSE) file for details.  
Please include a link to this repo if you use any of the dataset or code in your work and consider citing the following papers:
```
@article{Csaky:2022,
    title = {Generalizing Brain Decoding Across Subjects with Deep Learning},
    author = {Csaky, Richard and Van Es, Mats and Jones, Oiwi Parker and Woolrich, Mark},
    year = {2022},
    url = {https://arxiv.org/abs/2205.14102},
    journal	= {arXiv preprint arXiv:2205.14102},
    publisher = {arXiv}
}
@article{Csaky:2022,
    title = {Interpretable full-epoch multiclass decoding for M/EEG},
    author = {Csaky, Richard and Van Es, Mats and Jones, Oiwi Parker and Woolrich, Mark},
    year = {2022},
    url = {https://arxiv.org/abs/2205.14102},
    journal	= {arXiv preprint arXiv:2205.14102},
    publisher = {arXiv}
}
```
