fs = 1000;

%X = decimate(X, 2, 'fir');
% 
% X = X + randn(1, 586443)*2.5;
% X = ft_preproc_bandpassfilter(X, 500, [0.1 124.99], 5, 'but', 'onepass', 'reduce', 'plotfiltresp', 'yes');
% X = X(:,1:2:end);


%save(strcat('simulated_data/8event_snr1_500hz/resampled.mat'), 'X');




% X = resample(X, fs, 250);

% X = ft_preproc_highpassfilter(X, fs, 0.1, 5, 'but', 'onepass', 'reduce');
% X = ft_preproc_bandstopfilter(...
%     X, fs, [48 52], 5, 'but', 'onepass', 'reduce');
% X = ft_preproc_bandstopfilter(...
%     X, fs, [98 102], 5, 'but', 'onepass', 'reduce');

D = spm_eeg_load('mrc_data/spm/subject9');

D = osl_filter(D, [0.1, 124.999]);
D = osl_filter(D,-1*(50+[-0.2 0.2]));
D = osl_filter(D,-1*(100+[-0.3 0.3]));
D = osl_filter(D,-1*(150+[-0.4 0.4]));

%%
figure;
[Xpxx, Xf] = pwelch(D40(255,:,1), fs*2, fs, 2:0.1:fs*0.49, fs);
plot(Xf, Xpxx);

%%
figure;
[Xpxx, Xf] = pwelch(X(100, :), fs*2, fs, 2:0.1:125, fs);
plot(Xf, Xpxx);

%%
figure;
cwt(X(20, 1:50000), fs, 'TimeBandwidth', time_bandwidth);
%yline(30, 'white');
%yline(10, 'white');

%%
tr = 1000;
indices = [];
ind = 1;
for i=1:3:306
    [Xpxx, Xf] = pwelch(X(i, 1:200000), fs*2, fs, 0.1:0.1:3, fs);
    if Xpxx(1, 27) - Xpxx(1, 20) > tr
        indices(ind) = i;
        ind = ind+1;
    end
end

%%
tr = 0.5;
indices = [];
ind = 1;
for i=3:3:306
    [Xpxx, Xf] = pwelch(X(i, 1:200000), fs*2, fs, 0.1:0.1:40, fs);
    if Xpxx(1, 286) > tr
        indices(ind) = i;
        ind = ind+1;
    end
end

%%
pos = D.sensors('MEG').chanpos;
figure('Color','k','InvertHardCopy','off')
scatter3(pos(:,1),pos(:,2),pos(:,3),120,'b','filled')
hold on
for j = 1:3:size(pos,1)
	text(pos(j,1),pos(j,2),pos(j,3),num2str(j),'HorizontalAlignment','center','VerticalAlignment','middle','FontSize',12,'Color','w','FontWeight','bold');
end
axis equal
axis vis3d
axis off