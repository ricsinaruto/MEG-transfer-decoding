% script for Richard

%% simulate some binary data:

nclasses = 2;
ntrialsperclass = 50;
nchannels = 50;
Ttrial = 30; %timesteps within trial

rho = 0.025; % this parameter controls how separable the classes will be

% simulate evoked response over each condition:
clear ERF meantraj
for ich=1:nchannels
    ERF(:,ich) = conv(randn(Ttrial,1),[1,1,1,1,1,1,1,1,1]);
end
ERF = ERF - mean(ERF(1:3,:));
ERF = ERF(1:Ttrial,:);
for i=1:nclasses
    for ich=1:nchannels
        meantraj{i}(:,ich) = conv(randn(Ttrial,1),[1,1,1,1,1,1,1,1,1]);
    end
    meantraj{i} = rho*[meantraj{i}-mean(meantraj{i}(1:3,:))];
    meantraj{i} = meantraj{i}(1:Ttrial,:);
end

% now simulate each trial:
Sigma =wishrnd(eye(nchannels)/nchannels,nchannels);
data = [];
for iclass=1:nclasses
    for itrial=1:ntrialsperclass
        data = [data;ERF + meantraj{iclass} + mvnrnd(zeros(1,nchannels),Sigma,Ttrial)];
    end
end

% and specify the labels:
labels = repelem(eye(nclasses),ntrialsperclass*Ttrial,1);

% and trial timings:
T = Ttrial*ones(ntrialsperclass*nclasses,1);

%% simulate some more multi-class data:

nclasses = 4;
ntrialsperclass = 50;
nchannels = 50;
Ttrial = 30; %timesteps within trial

rho = 0.025; % this parameter controls how separable the classes will be

% simulate evoked response over each condition:
clear ERF meantraj
for ich=1:nchannels
    ERF(:,ich) = conv(randn(Ttrial,1),[1,1,1,1,1,1,1,1,1]);
end
ERF = ERF - mean(ERF(1:3,:));
ERF = ERF(1:Ttrial,:);
for i=1:nclasses
    for ich=1:nchannels
        meantraj{i}(:,ich) = conv(randn(Ttrial,1),[1,1,1,1,1,1,1,1,1]);
    end
    meantraj{i} = rho*[meantraj{i}-mean(meantraj{i}(1:3,:))];
    meantraj{i} = meantraj{i}(1:Ttrial,:);
end

% now simulate each trial:
Sigma =wishrnd(eye(nchannels)/nchannels,nchannels);
data = [];
for iclass=1:nclasses
    for itrial=1:ntrialsperclass
        data = [data;ERF + meantraj{iclass} + mvnrnd(zeros(1,nchannels),Sigma,Ttrial)];
    end
end

% and specify the labels:
labels = repelem(eye(nclasses),ntrialsperclass*Ttrial,1);

% and trial timings:
T = Ttrial*ones(ntrialsperclass*nclasses,1);

%% load real data
% load each channel, last is the label
dir = 'rich_data/participant1_pilot2/preproc20hz100hz_epoched/train_data_meg/';
data_ = [];
for i=0:306
    load(strcat(dir, 'cch', int2str(i), '.mat'));
    data_(i+1, :, :) = squeeze(cat(1,x_train_t,x_val_t));
end

% prepare real data
Ttrial_ = size(data_,3);
nclasses_ = 5;
ntrialsperclass_ = size(data_,2) / nclasses_;

labels_ = data_(307,:,:);
data_ = permute(data_(1:306,:,:), [3, 2, 1]);
data_ = reshape(data_, [], 306);

% prepare labels
T_ = Ttrial_*ones(ntrialsperclass_*nclasses_,1);
labels_ind = reshape(squeeze(labels_)', [], 1);
labels_ = zeros(size(labels_ind, 1), nclasses_);
for i=1:size(labels_ind, 1)
    labels_(i, labels_ind(i)+1) = 1;
end

%% replace simulation with real
data = data_;
labels = labels_;
T = T_;
Ttrial = Ttrial_;
nclasses = nclasses_;
nchannels = 306;
ntrialsperclass = ntrialsperclass_;


%% one vs all classification:
% options = [];
% options.classifier = 'SVM';
% options.NCV = 5; % number of cross val folds
% 
% acc_SVM = standard_classification(data,labels,T,options);

% maybe try some other classifiers:
% options = [];
% options.classifier = 'LDA';
% options.covtype = 'full';
% options.NCV = 5; % number of cross val folds
% 
% acc_LDA = standard_classification(data,labels,T,options); 

% and logistic regression:
% options = [];
% options.classifier = 'logistic';
% options.NCV = 5; % number of cross val folds
% 
% acc_LR = standard_classification(data,labels,T,options);

figure();plot(acc_SVM,'LineWidth',2); hold on;
%plot(acc_LDA,'LineWidth',2); 
%plot(acc_LR,'LineWidth',2); 
plot(1:Ttrial, 1/nclasses*ones(Ttrial,1),'k--');
xlabel('Time');ylabel('Classification accuracy');
%legend('SVM','LDA','Log Reg','Chance');
legend('SVM','Chance');

%% OR multiple pairwise:
options = [];
options.classifier = 'SVM';
options.pca = 80;
options.NCV = 5; % number of cross val folds
clear acc
for iclass1=1:nclasses
    for iclass2=iclass1+1:nclasses
        thisdata = data(labels(:,iclass1)==1 | labels(:,iclass2)==1,:);
        thislabels = labels(labels(:,iclass1)==1 | labels(:,iclass2)==1,iclass1);
        thisT = T(1:2*ntrialsperclass);
        acc(:,iclass1,iclass2) = standard_classification(thisdata,thislabels,thisT,options); 
        acc(:,iclass2,iclass1) = acc(:,iclass1,iclass2);
    end
end

meanacc = mean(acc(:,find(triu(ones(nclasses),1))),2);

figure();plot(meanacc,'LineWidth',2); hold on;
plot(1:Ttrial, 0.5*ones(Ttrial,1),'k--');
xlabel('Time');ylabel('Classification accuracy');
legend('SVM','Chance');


%% add in PCA:

options = [];
options.classifier = 'SVM';
options.NCV = 5; % number of cross val folds
options.pca = 5; % number of PC components to use

acc_SVM = standard_classification(data,labels,T,options);

figure();plot(acc_SVM,'LineWidth',2); hold on;
plot(1:Ttrial, 1/nclasses*ones(Ttrial,1),'k--');
xlabel('Time');ylabel('Classification accuracy');

%% or do with sliding window features:

options = [];
options.classifier = 'SVM';
options.NCV = 5; % number of cross val folds
options.pca = 80; % number of PC components to use
options.embeddedlags = [-5:1:5]; % number of timepoints to use as features

acc_SVM = standard_classification(data,labels,T,options);
acc_SVM = [nan(sum(options.embeddedlags<0),1);acc_SVM;nan(sum(options.embeddedlags>0),1)];
figure();plot(acc_SVM,'LineWidth',2); hold on;
plot(1:Ttrial, 1/nclasses*ones(Ttrial,1),'k--');
xlabel('Time');ylabel('Classification accuracy');

%% check cross-time generalisation matrices:

options = [];
options.classifier = 'SVM';
options.NCV = 10; % number of cross val folds
options.slidingwindow = 5; % number of timepoints to use as features
options.generalisationplot = true;

[acc_SVM,~,genplot] = standard_classification(data,labels(:,1),T,options);

figure();
subplot(1,2,1);
plot(acc_SVM,'LineWidth',2); hold on;
plot(1:Ttrial, 0.5*ones(Ttrial,1),'k--');
xlabel('Time');ylabel('Classification accuracy');

subplot(1,2,2);
imagesc(flipud(genplot))
colorbar;
xlabel('Training time');
ylabel('Testing time');


%% now try decoding on Fourier features:
clear freq_features

%setup stft params:
downsamplefactor = 1;
win = hamming(10,'periodic'); % a ten point hamming window
overlaplength = 9; % have these as overlapping as possible
Fs = 100; % sample rate

% note you may need to do this - osl comes with a function that overwrites
% the standard matlab functions we would use here:
rmpath('/Users/ricsi/Documents/GitHub/osl/spm12/external/fieldtrip/external/signal/')

        
% go trial by trial and channel by channel converting to frequency
% features:
for itrial = 1:length(T)
    tempdata = data([1:Ttrial]+(itrial-1)*Ttrial,:);
    clear tempdatatf
    for ich=1:nchannels
        [tempdatatf(:,:,ich),f] = stft(tempdata(:,ich),Fs,'Window',win,'OverlapLength',overlaplength);
    end
    nF = length(unique(abs(f)));
    tempdatatf = tempdatatf(find(f==0):end,:,:);
    tempdatatf = cat(2,zeros(nF,floor(length(win)/2),nchannels),tempdatatf);
    tempdatatf = cat(2,tempdatatf,zeros(nF,floor(length(win)/2)-1,nchannels));
    datatocat = permute(cat(3,real(tempdatatf),imag(tempdatatf)),[3,2,1]);
    timeseriestocat = [tempdata';zeros(size(tempdata'))]; %zeros are to align to imaginary part of TF transform
    freq_features(:,[1:Ttrial]+(itrial-1)*Ttrial,:) = cat(3,timeseriestocat,datatocat);
end
freq_features = permute(freq_features,[2,1,3]);

% sorry if this is a bit messy - but now freq_features is of dimension
% time x channels x frequency band. Note within the second dimension, you
% have the first N/2 components being the real Fourier coefficients, and
% the second N/2 components being the imaginary coefficients - use both in
% an SVM setup as below to get the best accuracy:

options = [];
options.classifier = 'SVM';
options.NCV = 5; % number of cross val folds
for ifreqband = 1:nF
    acc_SVM_freq(:,ifreqband) = standard_classification(freq_features(:,:,ifreqband),labels,T,options);
    leglabels{ifreqband} = [num2str(f(nF-2+ifreqband)),'Hz'];
end

figure();
plot(acc_SVM_freq,'LineWidth',2);
legend(leglabels);