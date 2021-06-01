fs = 250;

%X = decimate(X, 2, 'fir');

X = X + randn(1, 586443)*2.5;
X = ft_preproc_bandpassfilter(X, 500, [0.1 124.99], 5, 'but', 'onepass', 'reduce', 'plotfiltresp', 'yes');
X = X(:,1:2:end);


%save(strcat('simulated_data/8event_snr1_500hz/resampled.mat'), 'X');




% X = resample(X, fs, 250);

% X = ft_preproc_highpassfilter(X, fs, 0.1, 5, 'but', 'onepass', 'reduce');
% X = ft_preproc_bandstopfilter(...
%     X, fs, [48 52], 5, 'but', 'onepass', 'reduce');
% X = ft_preproc_bandstopfilter(...
%     X, fs, [98 102], 5, 'but', 'onepass', 'reduce');

