%% Load OSL
addpath(genpath('/Users/ricsi/Documents/GitHub/osl/osl-core'))
osl_startup

%% Set parameters
fs = 100;
band_pass = [0.1 124.99];
%freqs = [10 24 36 45];
freqs = [10 20];
%freqs = [8 11 14 17 20 23 26 29 35 38 41 45];
time_bandwidth = 120;
bin_width = 3;

%% Subject embedding correlation analysis
rho = corr(X');
%imagesc(rho);
%colormap('jet');
%colorbar;

[r3, y] = reord(rho, 2, 'true');

%% Apply forward filter
X = ft_preproc_bandpassfilter(...
    X, fs, band_pass, 5, 'but', 'onepass', 'reduce');

%% Save filtered data
save(strcat('simulated/8event_snr1/filtered125hz.mat'), 'X')

%% Plot wavelet and welch spectra of timeseries
figure;
cwt(squeeze(X(200, 1:10000)), fs, 'TimeBandwidth', time_bandwidth);
for i = 1:size(freqs, 2)
    yline(freqs(i), 'white');
end

%%
figure;
[Xpxx, Xf] = pwelch(x(:, :)', fs, fs/2, 3.0:0.1:50, fs);
plot(Xf, Xpxx);
%for i = 1:size(freqs, 2)
%    xline(freqs(i), 'red');
%end
%%
figure;
plot(X(1, 5000:6000));

%% Extract timeseries for the known frequencies
% compute wavelet to get timeseries for all frequencies
[wavelet, F] = cwt(X(1, 1000:end), fs, 'TimeBandwidth', time_bandwidth);

% find closest frequency from wavelet to the known frequencies
indices = [0];
for i = 1:size(freqs,2)
    [m, am] = min(abs(F-freqs(i)));
    indices(i) = am;
end

%% get stc from amax_stc
stc = zeros(size(amax_stc,2), size(freqs,2));
for i = 1:size(amax_stc,2)
    stc(i, amax_stc(1, i)+1) = 1;
end

%% Create state time course from known frequencies
stc = abs(wavelet(indices, :))';

% if power is below a threshold, set it to 0, to make stc less noisy
stc(stc<mean(stc)+std(stc)) = 0;

% normalize so that individual frequency powers add up to 1
stc = stc./(sum(stc, 2)+0.000001);

%%
figure;
area(stc(1:9000, :));

% get argmax of state time course
stc(:,9) = ones(size(stc, 1), 1) * 0.01;
[hh, amax_stc] = max(stc, [], 2);

%% Set color for area plots
figure;
a = area(stc(15000:17000, :));
a(1).FaceColor = [0.4, 0.0, 0.0];
a(2).FaceColor = [0.7, 0.0, 0.0];
a(3).FaceColor = [1.0, 0.0, 0.0];
a(4).FaceColor = [0.0, 0.4, 0.0];
a(5).FaceColor = [0.0, 0.7, 0.0];
a(6).FaceColor = [0.0, 1.0, 0.0];
a(7).FaceColor = [0.0, 0.0, 0.4];
a(8).FaceColor = [0.0, 0.0, 0.7];

%% Compute histogram of lifetimes for each state
for i = 1:8
    index = 1;
    distro = [0];
    for ts = 2:size(amax_stc, 1)
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
    
    distro(distro>1000) = 0;
    %distro = distro(distro>40);
    figure;
    histogram(distro, 'BinWidth', 5);
    xlim([0, 300]);
end

%% Train hmm on the extracted frequency timeseries
Xtrain = abs(wavelet(indices, :))';
options = [];
options.K = length(freqs);
options.zeromean = false;
options.order = 0;
[hmm, hmm_stc] = hmmmar(Xtrain, length(Xtrain), options);
