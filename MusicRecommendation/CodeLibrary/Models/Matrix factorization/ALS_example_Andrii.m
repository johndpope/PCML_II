%% input

clear all;
clc;

% Users
N = 1800; 
% Artists
D = 15000; 
% Real number of hidden factors for matrix generation
realK = 100; 
% Sparsity
S = floor(0.995 * D); 
% Number of hidder factors to use
K = 20;
% Maximum number of iterations for ALS
maxIter = 10;
% Regularization
lambda = 1;

% Ensuring reproducibility
seed = 1;
global RNDN_STATE  RND_STATE
RNDN_STATE = randn('state');
randn('state',seed);
RND_STATE = rand('state');
%rand('state',seed);
rand('twister',seed);

% Matrix generation
realU = randn(realK, N) + 2;
realM = randn(realK, D) + 2;
realR = realU' * realM;

% Selecting sparse submatrix
while(true)
    R = realR;
    for i=1:N
        idx = randperm(D);
        R(i,idx(1:S)) = 0;
    end
    % Ensuring that all rows and columns are non-empty
    if (min(sum(R > 0, 1)) > 0 && min(sum(R > 0, 2)) > 0)
        break;
    end
end

%% run
tic;
aveRating = zeros(1, D);

for i=1:D
    aveRating(i) = sum(R(:,i)) / sum(R(:,i) > 0);
end

M = [aveRating; randn(K - 1, D)];
U = zeros(K, N);


goodRows = cell(N, 1);
valRows = cell(N, 1);
for i=1:N
    goodRows{i} = find(R(i,:) > 0);
    valRows{i} = R(i, goodRows{i});
end
goodCols = cell(D, 1);
valCols = cell(D, 1);
for j=1:D
    goodCols{j} = find(R(:,j) > 0);
    valCols{j} = R(goodCols{j}, j);
end

[IR, JR] = ind2sub(size(R), find(R > 0));
Rvals = R(R > 0);
Rcur = zeros(length(IR), 1);

for iter=1:maxIter
    
    Rold = Rcur;
    
    for i=1:N
        I = goodRows{i};
        Ai = M(:,I) * M(:,I)' + lambda * length(I) * eye(K);
        Vi = M(:,I)  * valRows{i}';
        U(:,i) = Ai \ Vi; 
    end
    
    for j=1:D
        J = goodCols{j};
        Aj = U(:,J) * U(:,J)' + lambda * length(J) * eye(K);
        Vj = U(:,J) * valCols{j};
        M(:,j) = Aj \ Vj;
    end
    
    Rcur = sum(U(:,IR) .* M(:,JR))';    
    
     error = norm(Rvals - Rcur);
     change = max(abs(Rcur - Rold));
     fprintf('It = %03d Df = %0.3f Err = %0.3f\n', iter, change, error);
end
toc;

%% expected output
% 
% It = 001 Df = 467.620 Err = 57473.265
% It = 002 Df = 262.230 Err = 23825.653
% It = 003 Df = 47.984 Err = 17398.572
% It = 004 Df = 27.137 Err = 13891.180
% It = 005 Df = 20.802 Err = 11622.860
% It = 006 Df = 14.054 Err = 10022.027
% It = 007 Df = 13.889 Err = 8828.490
% It = 008 Df = 9.704 Err = 7902.409
% It = 009 Df = 8.224 Err = 7160.912
% It = 010 Df = 6.950 Err = 6552.774