# MEG-transfer-decoding
General PhD research code for modeling and decoding M/EEG data. Expect bugs and unexpected behaviour. This readme describes how to reproduce results in [Interpretable full-epoch multiclass decoding for M/EEG](https://arxiv.org/abs/2205.14102). We propose that a full-epoch multiclass model is better than sliding window and/or pairwise models for decoding visual stimuli, and we show how it can be used for multivariate pattern analysis (MVPA) through permutation feature importance (PFI).

**Since this repository is constantly evolving and contains a lot of other projects, make sure you are using the code version ([v0.1-paper](https://github.com/ricsinaruto/MEG-transfer-decoding/tree/v0.1-paper)) specifically created for the publication.**
```
git clone https://github.com/ricsinaruto/MEG-transfer-decoding --branch v0.1-paper
```

![Animation](https://github.com/ricsinaruto/MEG-transfer-decoding/blob/main/notebooks/mvpa_figures/animation.gif)

## Features
  :magnet: &nbsp; Train linear classifiers on MEG data.  
  :rocket: &nbsp; Can easily specify full-epoch/sliding-window, multiclass/pairwise mode, and extend to other classifiers.  
  :brain: &nbsp; Neuroscientifically interpretable features can be extracted and visualized from trained models.  
  :twisted_rightwards_arrows: &nbsp; Flexible pipeline with support for [multi-run modes](https://github.com/ricsinaruto/MEG-group-decode/edit/main/README.md#multi-run-modes) (e.g. cross-validation within and across subjects, different training set ratios, etc.)
  
## Table of Contents
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Examples](#examples)
- [Arguments](#arguments)
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
Set the function(s) you want to run to True in the ```func``` dictionary in ```args.py```.
Some relevant functions that are available:
* **train**: Trains a specified classifier on the specified dataset. This should only be used with a neural network model.
* **LDA_baseline**: Trains any non-neural network model (LDA, Logistic Regression, etc.) on the specified dataset for multiclass classification.
* **LDA_pairwise**: Same as ```LDA_baseline```, but the models are trained for pairwise classification across all pairs of classes.
* **LDA_channel**: Same as ```LDA_baseline```, but individual models are trained on individual channels of the MEG data, i.e. sliding-channel decoding.
* **PFIts**: runs temporal PFI for a trained model.
* **PFIch**: runs spatial or spatiotemporal PFI for a trained model, depending on input arguments
* **PFIfreq**: runs spectral PFI for a trained model.
* **PFIfreq_ch**: runs spatiospectral PFI for a trained model.
* **PFIfreq_ts**: runs temporospectral PFI for a trained model.
* **multi2pair**: Generates pairwise accuracies across all pairs of classes for a trained multiclass model (on the validation data).

Generally a model and dataset (specified in the corresponding variables in ```args.py```) is required to run anything. The dataset class used for this paper is always ```CichyData```.

### Multi-run modes
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
* ```SVM```: Implements the Support Vector Machine model, with the default kernel in sklearn.
* ```LDA_wavelet```: Same as ```LDA```, but run on the concatenated STFT features of the data.
* ```LDA_wavelet_freq```: Same as ```LDA_wavelet``` but run on a single frequency band from the STFT features.
* ```LogisticRegL1```: Logistic regression with L1 loss, basically a Lasso classifier.
* ```linearSVM```: Implements the linear SVM model from sklearn.
* ```QDA```: Implements the Quadratic Discriminant Analysis model.


## Examples
To replicate some of the results in the paper we provide args files in the examples folder. To try these out on the publicly available MEG data, follow these steps:  
1. ```python scripts/cichy_download.py``` to download data. (Use ```scripts/cichy92_download.py``` for the 92-class dataset)
2. ```python scripts/cichy_preproc_epoched.py``` to preprocess data. (Use ```scripts/cichy92_preproc_epoched.py``` for the 92-class dataset)
3. Copy the contents of the example args file you want to run into ```args.py```
4. ```python launch.py```

The following example args files are available (all args files are for the 118-class dataset, except when cichy92 is in the file name):
* ```args_nn.py```: trains the neural network model on full-epoch data.
* ```args_lda_pca_fe_multiclass.py```: trains the LDA-PCA multiclass model on full-epoch data.
* ```args_lda_pca_sw_multiclass.py```: trains the LDA-PCA multiclass sliding window model.
* ```args_lda_nn_fe_multiclass.py```: trains the LDA-NN multiclass model on full-epoch data.
* ```args_lda_nn_sw_multiclass.py```: trains the LDA-NN multiclass sliding window model.
* ```args_lda_nn_chn_multiclass.py```: trains separate LDA models on individual channels of the full-epoch data.
* ```args_lda_nn_fe_pairwise.py```: train the LDA-NN full-epoch model for pairwise classification.
* ```args_lda_nn_fe_multiclass2pairwise.py```: Loads a previously trained multiclass full-epoch LDA-NN model and computes pairwise accuracies on the validation data.
* ```args_lda_nn_fe_multiclass_temporalPFI.py```: Loads a previously trained multiclass full-epoch LDA-NN model and runs temporal PFI.
* ```args_lda_nn_fe_multiclass_spatialPFI.py```: Loads a previously trained multiclass full-epoch LDA-NN model and runs spatial PFI.
* ```args_lda_nn_fe_multiclass_spatiotemporalPFI.py```: Loads a previously trained multiclass full-epoch LDA-NN model and runs spatiotemporal PFI.
* ```args_lda_nn_fe_multiclass_spectralPFI.py```: Loads a previously trained multiclass full-epoch LDA-NN model and runs spectral PFI.
* ```args_lda_nn_fe_multiclass_temporospectralPFI.py```: Loads a previously trained multiclass full-epoch LDA-NN model and runs temporospectral PFI.
* ```args_cichy92_lda_pca_fe_multiclass.py```: trains the LDA-PCA multiclass model on full-epoch data of the 92-class dataset.

Steps 1 and 2 can be skipped if running on non-public data. The relevant data paths in the args file have to modified in this case. Note that results from running the examples will not 100% reproduce our results, because we used the raw continuous MEG data. Also, different random seeds may cause (very) small differences.

Please note that the variable "n" in the args files can be used to control the number of subjects. This is set to 1 by default for just trying out the methods. If you want to run on all 15 subjects just change this to 15 in one of the example args files and the whole pipeline will be run on each subject automatically.  

When running multiple trainings or analyses on the same dataset the ```load_data``` and ```num_channels``` arguments should be set as described in the [Arguments](#arguments) section, for faster data loading. This is already included in the example args files running PFI.  

Modifying these scripts to, for example, run on the 92-class data is easy as only the ```data_path``` and ```num_classes``` has to be modified as demonstrated by the ```args_cichy92_lda_pca_fe_multiclass.py``` script. If other [models](#models) are desired, such as SVM, only the ```model``` arguments needs to be changed. Further arguments and their behaviour can be explored in the [Arguments](#arguments) section, such as window size, pca dimension, number of classes, etc.

### Visualizations
After running some the example scripts in the previous section, one can use the ```notebooks/mvpa_paper_tutorial.ipynb``` jupyter notebook to produce most of the figures in the paper. The paths for some of the figures are pre-set so that it seamlessly loads the results from the examples scripts, while for other figures manually setting the correct paths might be required.


## Arguments
This section describes the behaviour of some arguments in the ```args.py``` file.  
### Experiment arguments:  
* ```load_model```: Path(s) to load a trained model file from. Can be a single path or a list of path(s) when using multi-run mode, pointing to for example different subjects. When running PFI analysis this argument should be specified. If this is None, then a new model will be initialized based on the ```model``` argument.
* ```result_dir```: Path(s) where model and other analysis outputs should be saved. Behaviour is similar to ```load_model```. Cannot be empty.
* ```model```: Model class to use in training/analysis. Please see the [Models](#models) section for possible classes. Can only be empty when using the ```load_model``` argument.
* ```dataset```: Class of dataset to use. For reproducing results in the paper this should always be CichyData. Other datasets pertain to other projects.
* ```max_trials```: Ratio of training data to use. 1 means use all the data. Can be useful for exploring how training data size affects performance. Can be a list of values to run multiple training with different training data ratios in a single call.  

### Neural Network arguments:
* ```learning_rate```: Learning rate of the Adam optimizer.
* ```batch_size```: Batch size for training and validation data.
* ```epochs```: Number of full passes over the training data.
* ```val_freq```: Frequency of running a pass over validation data (in epochs).
* ```print_freq```: Frequency of printing training metrics (in epochs).
* ```units```: List of integers, where each value is the size of a hidden layer in the neural network.
* ```activation```: Activation function between layers. Can be set to any torch function. Use ```torch.nn.Identity()``` for a linear neural network.
* ```p_drop```: Dropout probability between layers.

### Classification (task) arguments:
* ```sample_rate```: Start and end timesteps for cropping trials. [0, -1] would select the full trial.
* ```num_classes```: 118 for the 118-class dataset, or 92 for the 92-class dataset. Alternatively if one wants to use only a subset of the conditions, this can be set to a lower number which will automatically only load the first x classes.
* ```dim_red```: This is either the number of components used in LDA-PCA, or the size of the dimensionality reduction layer in the neural network (and in LDA-NN).
* ```load_conv```: Path(s) to trained neural network from which the dimensionality layer will be extracted for use in LDA-NN. Can be a list to facilitate running over multiple subjects. If False, LDA-PCA will be run. A third mode can be achieved by setting this to a non-existent path, whereby no dimensionality reduction or pca will be applied to the data, and thus LDA is run on the raw data.
* ```halfwin```: This is half the window size for sliding-window models, e.g. at a sampling rate of 100Hz set this to 5, to get a 100ms window. Importantly, if full-epoch modeling is desired this should be set to half the size of the ```sample_rate``` range. This parameter has two more uses depending on running mode. When running temporal PFI it controls the window size for permuting (similar to sliding-window decoding). When running spectral PFI it controls the frequency band width, so should be set to 0 if maximum frequency sensitivity is desired.

### Dataset arguments:
* ```data_path```: Path(s) to subject folders. Can be list to facilitate running over multiple subjects. Importantly *sub* should be present in each path, otherwise the folder is ignored. Each subject folder should contain subfolders for each condition (cond0, cond1, ...), and each condition folder should contain the individual trials for that condition (trial0.npy, trial1.npy, ...), where each trial is a numpy array of size (timesteps, channels).
* ```num_channels```: This specifies the channel indices to be used. If all channels are desired, and when first running over a dataset this should be set to the number of MEG channels, e.g. ```list(range(306))```. In subsequent runs, when used in combination with the ```load_data``` argument, ```list(range(307))```: should be used to account for the target variable. This is very important as otherwise the classification won't work.
* ```shuffle```: Whether to shuffle order of trials within each split.
* ```whiten```: Number of pca components used for whitening the data. if False no whitening is performed.
* ```split```: Specifies the start and end of the validation split (as a ratio). Can be a list to automatically run cross-validation.
* ```sr_data```: The desired sampling rate of the data, e.g. 100 (Hz).
* ```original_sr```: The original sampling rate of the data that is processed, e.g. 1000 (Hz).
* ```save_data```: Whether to save the downsampled and split data so that it can be used with the ```load_data``` argument in subsequent runs for faster processing.
* ```dump_data```: Path(s) to save the downsampled and split data.
* ```load_data```: This should be an empty string when first using a dataset. On subsequent runs it can be set to the ```dump_data``` path (if ```save_data``` was True), for faster loading. Don't forget to increase the number of channels by 1 in ```num_channels```.

### PFI arguments:
* ```closest_chs```: Path to a file containing a list of channel indices and their closest neighbours. This is used for spatial PFI. examples/closest1 is given for running PFI on individual channels, and examples/closest4 is given for running PFI on 4-channel spatial windows.
* ```PFI_inverse```: This is normally False. As described in the supplementary materials of the paper the PFI method can be inverted if desired.
* ```pfich_timesteps```: List, where each element specifies a time-window for spatiotemporal PFI. For normal spatial PFI this should be set to [[0, num_timesteps]].
* ```PFI_perms```: Number of times to compute PFI with different permutations.
* ```PFI_val```: Whether to run PFI on validation or training data. Normally set to True.


## Contributing
This is an active research repository and contains multiple projects in addition to the code described in this readme relevant to the paper. Thus the pipeline is very general and easily extendable to other models/datasets. For example implementing other sklearn classifiers is easy by just subclassing the ```LDA``` class and updating the ```init_model``` function with the relevant classifier.  

Other neural networks can be used, with the only requirement being that they have a ```loss``` function which accepts the specified arguments and returns a dictionary of losses/accuracies. The ```__init__``` function should accept an ```Args``` object which stores all arguments. We recommend sublcassing the ```SimpleClassifier``` class.  

Dataset classes for other datasets can be implemented, however CichyData is very general and will work with any EEG/MEG epoched data that is saved in the required format (see the ```data_path``` argument).

Additional behaviour can be easily achieved and controlled by creating extra arguments in the ```Args``` class, which then are accessible in the dataset and model classes. Additional functionality can be implemented by registering a new function at the end of ```training.py```. This can then be accessed from the ```func``` dictionary in ```Args```.

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
