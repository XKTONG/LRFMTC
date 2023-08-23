close all;
clc;
clear;
addpath(genpath(pwd));

randn('state',num); rand('state',num);

% the synthetic data generation

DIM = [50,50,50];     % Dimensions of data
R1 = 16;     % R = 40;
R2 = 16;
R3 = 16;
% 
VV{3} = 1*eye(R3) + diag(randperm(R3));
VV{1} = 1*eye(R1) + diag(randperm(R1)) ;          
VV{2} = 1*eye(R2) + diag(randperm(R2)) ;

U{1} = orth(randn(DIM(1),R1));        
U{2} = orth(randn(DIM(2),R2));
U{3} = orth(randn(DIM(3),R3));

core = ttm(tensor(randn(R1,R2,R3)),VV);
X = double(ttensor(core,U)); 
X = (prod(DIM)/sqrt(sum(X(:).^2)))*X; 


%% Random missing values
ObsRatio = 0.2;    % Observation rate: [0 ~ 1] 
DIM = size(X);
[~,Omega] = sort(rand(1,prod(DIM))); 
Omega = Omega(1:round(ObsRatio*prod(DIM)));
O = zeros(DIM); 
O(Omega) = 1;
nObs = sum(O(:));
mask = O;
disp(nObs/numel(X))

%% Add noise
SNR = 20;                     % Noise levels
sigma2 = var(X(:))*(1/(10^(SNR/10)));
GN = sqrt(sigma2)*randn(DIM);
%GN = zeros(DIM);

%% Generate observation tensor Y
Y = X + GN;
Y = O.*Y;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FPC
ts = tic;
[est_X,estR,rses] = Fast_LRFMTC(Y,X,60);
t_total(num) = toc(ts);

RR(num,1) = estR(1);
RR(num,2) = estR(2);
RR(num,3) = estR(3);
    
err = est_X(:) - X(:);
rrse = sqrt(sum(err.^2)/sum(X(:).^2));
RSE(num) = rrse;
RSEmin(num) = min(rses);
 
  






