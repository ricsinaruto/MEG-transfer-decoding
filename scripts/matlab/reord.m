function [r3, y]=reord(A,power,do_plots)
% REORD.M  Try Spectral reordering on Oxford data.
%
% [r3]=reord(A,power,do_plots)
% Based on spec_cat.m
%
%
% DJH, August 2003

if nargin<3
    do_plots=0;
end

%%%%%% Spectral Ordering %%%%%%
%A=sparse(A);
G = (A + A')/2;             %forces symmetry
G = G.^power;
Q = -G;
Q = triu(Q,1) + tril(Q,-1);
Q = Q - diag(sum(Q));

t = 1./sqrt((sum(G)));
Q =  diag(t)*Q*diag(t);    %Normalized Laplacian  

% get second eigenvalue

%Qs=sparse(Q);
%[V,D]=eigs(Qs,2,'SM');

[V,D] = eig(Q);


d = diag(D);

[a,b] = sort(d);
index = b(2);
%keyboard
v2 = V(:,index);

v2scale = diag(t)*v2;

[y,r3] = sort(v2scale);




%save r3

if do_plots
    figure
    imagesc(A')
    colorbar
    title('Original','FontWeight','Bold')

    figure
    colormap('default')
    imagesc(A(r3,r3)')
    title('Reordered','FontWeight','Bold')
    colorbar
    figure;plot(y);
end
%keyboard
%display('orig twosum = '), twosum(A)

%display('new twosum = '),  twosum(A(r3,r3))





