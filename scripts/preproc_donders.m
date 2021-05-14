% Pre-proc steps
% 1) Import the raw data (.ds folders)
% 2) Filter to Frequency band(s) of interest (e.g. 0.1-50Hz)
% 3) Add notch to filter to mains noise (50 or 60Hz, and maybe 100/120 Hz)
% 4) Downsample the data (bearing in mind Nyquist)
% 5) Bad channel/bad segment detection
% 6) Run ICA
% 7) Sanity check the data *now*. Plot an evoked/induced response
% In other scenarios, do source recon and coreg, etc.

%% Load OSL
addpath(genpath('/Users/ricsi/Documents/GitHub/osl/osl-core'))
osl_startup

%% 1. Set parameters and import data
high_pass = 0.1;  % high pass frequency
low_pass = 50;  % low pass frequency
sampling_rate = 250;  % sampling rate for downsampling
bad_significance = 0.2;  % threshold for bad segment removal
data_dir = 'sub-A2011/';
ds_dir = strcat(data_dir, 'sub-A2011_task-rest_meg.ds');

D = osl_import(ds_dir, 'artefact_channels', 'EEG');

% set EOG and ECG channels
D = D.chantype(find(strcmp(D.chanlabels, 'EEG058')), 'EOG');
D = D.chanlabels(find(strcmp(D.chanlabels, 'EEG058')), 'EOG');
D = D.chantype(find(strcmp(D.chanlabels, 'EEG059')), 'ECG');
D = D.chanlabels(find(strcmp(D.chanlabels, 'EEG059')), 'ECG');

%% Look at the data
D = oslview(D);

%% 2-5. Filter and downsample
% based on looking at the data in previous step, we crop the end artefact
crop = 8;  % how many seconds to remove from the end
start = D.nsamples / D.fsample - crop;
BadEpochs = {[start, start+crop]};
D = set_bad_events(D, BadEpochs, 'MEGGRAD', false);
D = set_bad_events(D, BadEpochs, 'EOG', false);
D = set_bad_events(D, BadEpochs, 'ECG', false);

% bandpass and notch filters
start = D.nsamples - crop * D.fsample + 10;
D(:, 1:start, 1) = osl_filter(D(:, 1:start, 1), [high_pass low_pass],...
                              'prefix', '', 'fs', D.fsample);
D(:, 1:start, 1) = osl_filter(D(:, 1:start, 1), -[48 52],...
                              'prefix', '', 'fs', D.fsample);
D(:, 1:start, 1) = osl_filter(D(:, 1:start, 1), -[98 102],...
                              'prefix', '', 'fs', D.fsample);

% downsample
D = spm_eeg_downsample(struct('D', D, 'fsample_new', sampling_rate));

% currently we only remove bad segments, not bad channels
%D = osl_detect_artefacts(D, 'badtimes', false, 'modalities', {'MEGGRAD'});
D = osl_detect_artefacts(D, 'badchannels', false,...
                            'event_significance', bad_significance,...
                            'modalities', {'MEGGRAD'});
D.save();

%% Look at the data
D = oslview(D);

%% 6. ICA artefact removal
% remove very beginning because of filtering
BadEpochs = {[0, 0.1]};
D = set_bad_events(D, BadEpochs, 'MEGGRAD', false);

% remove identified bad segments from eog and ecg
BadEpochs = {};
ev = D.events;
i = 1;
for j = 1:numel(ev)
    if strcmp(ev(j).value,'MEGGRAD')
        BadEpochs{i} = [ev(j).time, ev(j).time + ev(j).duration];
        i = i + 1;
    end
end

D = set_bad_events(D, BadEpochs, 'EOG', true);
D = set_bad_events(D, BadEpochs, 'ECG', true);

% run ICA and visualize for manual artefact rejection
D = osl_africa(D, 'do_ica', true,...
                  'do_ident', 'auto',...
                  'do_remove', true,...
                  'used_maxfilter', false);
D = osl_africa(D, 'do_ica', false, 'do_ident', 'manual');
D.save();

%% Look at the data
D = oslview(D);

%% Save data and good indices
D = D.montage('switch', 2);
chs = find(strcmp(D.chantype, 'MEGGRAD'));
chs(D.badchannels) = [];
X = D(chs, :, 1);
save(strcat(data_dir, 'preprocessed_data_new.mat'), 'X')

% save good indices so we know which timesteps to remove
X = ~D.badsamples(31, :, :);
save(strcat(data_dir, 'good_samples_new.mat'), 'X')


%% Legacy code
% set to zero cropped portions for visualization
% start = D.nsamples - crop * D.fsample;
% early = 0.1 * D.fsample + 1;
% D(:,start:start + crop * D.fsample,:) = 0;
% D(:,1:early-1,:) = 0;

% normalize for visualization
% chs = find(strcmp(D.chantype,'MEGGRAD'));
% D(chs, early:start-1, 1) = normalize(D(chs, early:start-1, 1), 2);
% chs = find(strcmp(D.chantype,'EOG'));
% D(chs, early:start-1, 1) = normalize(D(chs, early:start-1, 1), 2);
% chs = find(strcmp(D.chantype,'ECG'));
% D(chs, early:start-1, 1) = normalize(D(chs, early:start-1, 1), 2);
