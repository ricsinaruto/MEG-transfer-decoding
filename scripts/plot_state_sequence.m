%% Plots in solid colors the cumulative state distribution over time
function plot_state_sequence(Gamma, Fs, cmapind)
    if nargin<2
        Fs=1;
    end
    [T,K] = size(Gamma);

    GamCum = cumsum(Gamma,2);
    GamCum = [zeros(T,1),GamCum];
    if nargin<3 || cmapind==1
        colors = flipud(parula(K));
    elseif cmapind==2
        colors = flipud(winter(K));
    elseif cmapind==3
        colors = flipud(hot(K));
    elseif cmapind==4
        colors = flipud(copper(K));
    elseif cmapind==5
        temp = utils.set1_cols;
        for i=1:length(temp)
            colors(i,:) = temp{i};
        end
    end
    colors(13,:) = [1,1,1];
    for k=2:K+1
        %plot([1:T]*Fs,GamCum(:,k),'LineWidth',2,'Color',colors(k,:));
        polyshape = [GamCum(:,k-1);flipud(GamCum(:,k))];
        patch([[1:T],[T:-1:1]]*Fs,polyshape,colors(k-1,:));
        hold on;
    end
    ylim([0,1]);
    xlim([1,T]*Fs);
    xlabel('Time')
    ylabel('State probability')
end