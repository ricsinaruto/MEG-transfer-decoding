%% 1. Set parameters and import data
high_pass = 0.1;  % high pass frequency
low_pass = 124.999;  % low pass frequency
data_dir = 'rich_data/RC/';
fif = strcat(data_dir, 'RC_task.fif');

matlabbatch{1}.spm.meeg.convert.dataset = {'/Users/ricsi/Documents/GitHub/MEG-transfer-decoding/scripts/rich_data/RC/task_part1_rc_raw_tsss_mc.fif'};
matlabbatch{1}.spm.meeg.convert.mode.continuous.readall = 1;
matlabbatch{1}.spm.meeg.convert.channels{1}.all = 'all';
matlabbatch{1}.spm.meeg.convert.outfile = '';
matlabbatch{1}.spm.meeg.convert.eventpadding = 0;
matlabbatch{1}.spm.meeg.convert.blocksize = 3276800;
matlabbatch{1}.spm.meeg.convert.checkboundary = 1;
matlabbatch{1}.spm.meeg.convert.saveorigheader = 0;
matlabbatch{1}.spm.meeg.convert.inputformat = 'autodetect';

%% Load D object
D = spm_eeg_load('/Users/ricsi/Documents/GitHub/MEG-transfer-decoding/scripts/rich_data/RC/task_part1_rc_raw_tsss_mc');

%%
D = osl_import('/Users/ricsi/Documents/GitHub/MEG-transfer-decoding/scripts/rich_data/RC/task_part1_rc_raw_tsss_mc.fif');

%% bandpass filter
D = osl_filter(D, [high_pass low_pass], 'fs', D.fsample);

%% notch filter
D = osl_filter(D, -[48 52], 'fs', D.fsample);
D = osl_filter(D, -[98 102], 'fs', D.fsample);

%% Look at the data
D = oslview(D);

%% plot spectrum of all channels
figure;
for i=1:3:306
    [Xpxx, Xf] = pwelch(D(i, 1:300000, 1), fs, fs/2, 1:0.1:200, fs);
    plot(Xf, Xpxx);
    hold on;
end

%% Coregistration using SPM
S = [];
S.D = D;
S.mri = fullfile('MNI152_T1_1mm.nii.gz');
S.useheadshape = 1;
S.use_rhino = 1;
S.forward_meg = 'Single Shell';
S.fid.label.nasion = 'Nasion';
S.fid.label.lpa = 'LPA';
S.fid.label.rpa = 'RPA';
D_head=osl_headmodel(S);

%%
rhino_display(D_head);

%% epoching

% define the trials we want from the event information
S2 = [];

S2.D = D;
D_continuous=spm_eeg_load(S2.D);

pretrig = -100; % epoch start in ms
posttrig = 1000; % epoch end in ms
S2.timewin = [-100 1000];

trialdef = [];
events = [256 512 1024 2048];
trialdef(1).conditionlabel = '256';
trialdef(1).eventtype      = 'STI101_up';
trialdef(1).eventvalue     = 256;
    
S2.reviewtrials = 0;
S2.save = 0;

[epochinfo.trl, epochinfo.conditionlabels] = spm_eeg_definetrial(S2);

%%%%
% adjust timings to account for delay between trigger and visual display
if(opt.epoch.timing_delay~=0)
    timing_delay=opt.epoch.timing_delay; % secs
    epochinfo_new=epochinfo;
    epochinfo_new.trl(:,1:2)=epochinfo.trl(:,1:2)+round(timing_delay/(1/D_continuous.fsample));
    epochinfo=epochinfo_new;
end

%%%%
% do epoching
S3 = epochinfo;
S3.D = D_continuous;
S3.bc = 0;
D = osl_epoch(S3);

spm_files_epoched_basenames{subnum}=['e' spm_files_basenames{subnum}];

opt_results.spm_files_epoched_basename=spm_files_epoched_basenames{subnum};
epoched=true;

% do plot of trial timings
plot_name_prefix='opt-epoch: ';
figs = report.trial_timings(D, [], plot_name_prefix);
opt_report=osl_report_set_figs(opt_report,arrayfun(@(x) sprintf('opt-epoch_%s',get(x,'tag')),figs,'UniformOutput',false),figs,arrayfun(@(x) get(x,'name'),figs,'UniformOutput',false));
opt_report=osl_report_print_figs(opt_report); 