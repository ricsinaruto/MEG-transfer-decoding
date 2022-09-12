dir_temp = '/Users/ricsi/Documents/GitHub/MEG-transfer-decoding/scripts/replay_data/correct_data/sess1/highpass_0.5.opt';

data= nan(1,1,1,1); % channel*time points(-3s - 1s)*trials*sessions

% load the epoched data
opt=load(fullfile(dir_temp, 'opt.mat'));
opt=opt.opt;
DLCI=spm_eeg_load(fullfile(dir_temp,[opt.results.spm_files_epoched_basenames{1,1}]));

% load good channels:
chan_MEG = indchantype(DLCI,'meeg');

% load good trials
allconds=[{'S1'},{'S2'},{'S3'},{'S4'},{'S5'},{'S6'},{'S7'},{'S8'}];

all_trls = sort(DLCI.indtrial(allconds(:)));
good_trls = sort(DLCI.indtrial(allconds(:),'good'));
good_ind=ismember(all_trls,good_trls);

% get clean data:
cldata = DLCI(chan_MEG,:,good_trls);

% might be needed to select channels (+find_goodchannel.m)
%chan_inds = indchantype(SessionMEG,'meeg','GOOD');
%Channindex  = indchantype(SessionMEG,'meeg','ALL');
%[~,ind]=setdiff(Channindex,chan_inds);

