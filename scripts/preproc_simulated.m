%% Load OSL
addpath(genpath('/Users/ricsi/Documents/GitHub/osl/osl-core'))
osl_startup

%% Set parameters and create D object for OSL
high_pass = 0.1;
low_pass = 50;
bad_significance = 0.2;
fsample = 250;

data_dir = 'simulated';

s.type = 'continuous';
s.Nsamples = size(X, 2);
s.Fsample = fsample;
s.timeOnset = 0;
s.data = file_array('data.mat', [1, s.Nsamples]);
s.fname = 'dsub-data.mat';
s.path = strcat('/Users/ricsi/research/donders/', data_dir);

channels.type = 'MEGGRAD';
channels.bad = 0;
channels.label = 'SIM';
channels.units = 'fT';
s.channels = channels;

trials.label = 'Undefined';
trials.events = {};
trials.onset = 0;
trials.bad = 0;
trials.tag = [];
trials.repl = 1;
s.trials = trials;

D = meeg(s);

%% Look at the data
D = oslview(D);

%% Bandpass and notch filters
D = osl_filter(D, [high_pass low_pass], 'prefix', '');
D = osl_filter(D, -[48 52], 'prefix', '');
D = osl_filter(D, -[98 102], 'prefix', '');

% remove bad segments like in real data
%D = osl_detect_artefacts(D, 'badtimes', false, 'modalities', {'MEGGRAD'});
D = osl_detect_artefacts(D, 'badchannels', false,...
                            'event_significance', bad_significance,...
                            'modalities', {'MEGGRAD'});


%% Look at the data
D = oslview(D);

%% Save processed data and good indices
X = D(1, :, 1);
save(strcat('preprocessed_data_new.mat'), 'X')

X = ~D.badsamples(1,:,:);
save(strcat('good_samples_new.mat'), 'X')
