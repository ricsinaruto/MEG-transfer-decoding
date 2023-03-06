%% Load OSL
addpath(genpath('/Users/ricsi/Documents/GitHub/osl/osl-core'))
osl_startup


%% Load file
%load('../../results/cichy_quantized/data/matlab100hz.mat');
%load('../../results/cichy_quantized/AR255/100hz.mat');
load('../../results/cichy_quantized/wavenet/generated_datarecursivetop-p75%.mat');
x = double(X);
X_std = std(x');

%% mulaw_inv
mu = 255;
x = (x - 0.5) / mu * 2 - 1;
x = sign(x) .* (exp(log(1 + mu) .* abs(x)) - 1) / mu;


%% PCA and normalization
[coeff, X] = pca(X', 'NumComponents', 80);
X = (X - mean(X))./std(X);


%% Train hmm on pca data
options = [];
options.K = 8;
options.Fs = 100;
options.verbose = 1;
options.order = 0; % 0 means gaussian model

[hmm, Gamma, ~, vpath] = hmmmar(X, length(X), options);
save('hmm_cichy.mat', 'hmm', '-v7.3');


%% Analyse hmm output
