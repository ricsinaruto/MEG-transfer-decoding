% figure;
% plot(0:0.1:300,y, 'red', 'LineWidth', 2);
% set(gcf, 'color', 'none');    
% set(gca, 'color', 'none');
% exportgraphics(gcf,'transparent.eps',...   % since R2020a
%     'ContentType','vector',...
%     'BackgroundColor','none')

figure;
x = 1:1:276;

for i=1:100
    plot(squeeze(X(7,i,:)+i*6));
    hold on;
end
