% Retrieve eyetracker data from the raw MEG data. From the eyedata those
% trials are selected that are present in the processed MEG data. X- and Y-
% gaze positions on the screen (in pixels) are transformed to X- and Y-
% positions in visual degrees, relative to the central fixation point.
% Eyedata is resampled at 600 Hz and time locked both to stimulus onset and
% stimulus change.
%
% INPUT
%   subj (int): subject ID, ranging from 1 to 33, excluding 10.
%
% OUTPUT
%   saves result on disk.
%   data (struct): eye-data containing 3 channels (x and y gaze, pupil
%       diameter (a.u.)). Time locked to stimulus onset.
%   data_shift (struct): eye-data containing 3 channels (x and y gaze, pupil
%       diameter (a.u.)). Time locked to stimulus change.


%% change X and Y positions into visual angle relative to fixation
screenRight=1919;
screenLeft=0;
screenTop=0;
screenBottom=1079;

% range of voltages and recording range as defined in FINAL.INI
minVoltage=-5;
maxVoltage=5;
minRange=0;
maxRange=1;


voltageHor=D(382,:,1);
voltageVer=D(383,:,1);

% see eyelink1000_UserManual for formulas:
RangeHor = (voltageHor-minVoltage)./(maxVoltage-minVoltage); % voltage range proportion
S_h = RangeHor.*(maxRange-minRange)+minRange; % proportion of screen width or height

R_v = (voltageVer-minVoltage)./(maxVoltage-minVoltage);
S_v = R_v.*(maxRange-minRange)+minRange;

xGaze = S_h.*(screenRight-screenLeft+1)+screenLeft;
yGaze = S_v.*(screenBottom-screenTop+1)+screenTop;

%% edfmex
edfstruct = edfmex('DISPEYE_2021_11_30_11_45.EDF');

%% plotting


