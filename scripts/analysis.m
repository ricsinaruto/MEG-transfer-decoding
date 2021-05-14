%% Load OSL
addpath(genpath('/Users/ricsi/Documents/GitHub/osl/osl-core'))
osl_startup

%% Set parameters
fs = 250;
band_pass = [0.1 124.99];
%freqs = [10 24 36 45];
freqs = [10 14 18 22 26 33 38 45];
%freqs = [8 11 14 17 20 23 26 29 35 38 41 45];
time_bandwidth = 120;
bin_width = 3;

%% Apply forward filter
X = ft_preproc_bandpassfilter(...
    X, fs, band_pass, 5, 'but', 'onepass', 'reduce');

%% Save filtered data
save(strcat('simulated/8event_snr1/filtered125hz.mat'), 'X')

%% Plot wavelet and welch spectra of timeseries
figure;
cwt(X(1, 20000:29000), fs, 'TimeBandwidth', time_bandwidth);
for i = 1:size(freqs, 2)
    yline(freqs(i), 'white');
end

figure;
[Xpxx, Xf] = pwelch(X(1, 1000:end), fs*2, fs, 1:0.1:100, fs);
plot(Xf, Xpxx);
for i = 1:size(freqs, 2)
    xline(freqs(i), 'red');
end

figure;
plot(X(1, 1:3000));

%% Extract timeseries for the known frequencies
% compute wavelet to get timeseries for all frequencies
[wavelet, F] = cwt(X(1, 1000:end), fs, 'TimeBandwidth', time_bandwidth);

% find closest frequency from wavelet to the known frequencies
indices = [0];
for i = 1:size(freqs,2)
    [m, am] = min(abs(F-freqs(i)));
    indices(i) = am;
end

%% Create state time course from known frequencies
stc = abs(wavelet(indices, :))';

% if power is below a threshold, set it to 0, to make stc less noisy
stc(stc<mean(stc)+std(stc)) = 0;

% normalize so that individual frequency powers add up to 1
stc = stc./(sum(stc, 2)+0.000001);

figure;
area(stc(1:9000, :));

% get argmax of state time course
stc(:,9) = ones(size(stc, 1), 1) * 0.01;
[hh, amax_stc] = max(stc, [], 2);

%% Compute histogram of lifetimes for each state
for i = 1:8
    index = 1;
    distro = [0];
    for ts = 2:size(stc, 1)
        % end
        if amax_stc(ts-1, 1) == i && amax_stc(ts, 1) ~= i
            distro(index) = ts - distro(index);
            index = index + 1;
        end

        % start
        if amax_stc(ts, 1) == i && amax_stc(ts-1, 1) ~= i
            distro(index) = ts;
        end
    end
    
    figure;
    histogram(distro, 'BinWidth', bin_width);
end

%% Train hmm on the extracted frequency timeseries
Xtrain = abs(wavelet(indices, :))';
options = [];
options.K = length(freqs);
options.zeromean = false;
options.order = 0;
[hmm, hmm_stc] = hmmmar(Xtrain, length(Xtrain), options);
