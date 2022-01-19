% script for Richard
addpath(genpath('../../osl/osl-core'))
osl_startup;


%% load real data
% load each channel, last is the label
dir = '/gpfs2/well/woolrich/projects/disp_csaky/subj1_pilot2/preproc_epoched/train_data_meg/';
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
ntrialsperclass = ntrialsperclass_;



%% OR multiple pairwise:
options = [];
options.classifier = 'SVM';
options.pca = 80;
options.NCV = 5; % number of cross val folds
options.embeddedlags = [-20:1:20]; % number of timepoints to use as features
clear acc
for iclass1=1:nclasses
    for iclass2=iclass1+1:nclasses
        thisdata = data(labels(:,iclass1)==1 | labels(:,iclass2)==1,:);
        thislabels = labels(labels(:,iclass1)==1 | labels(:,iclass2)==1,iclass1);
        thisT = T(1:2*ntrialsperclass);
        acc_SVM = standard_classification(thisdata,thislabels,thisT,options);
        acc_SVM = [nan(sum(options.embeddedlags<0),1);acc_SVM;nan(sum(options.embeddedlags>0),1)];
        acc(:,iclass1,iclass2) = acc_SVM;
        acc(:,iclass2,iclass1) = acc(:,iclass1,iclass2);
    end
end

meanacc = mean(acc(:,find(triu(ones(nclasses),1))),2);

figure();plot(meanacc,'LineWidth',2); hold on;
plot(1:Ttrial, 0.5*ones(Ttrial,1),'k--');
xlabel('Time');ylabel('Classification accuracy');
legend('SVM','Chance');

