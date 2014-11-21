% Random algorithm
% Generates 0 with probability p
% or number between 1 and maxValue uniformly
% Used for STRONG generalization - predict for new users
% required varargin P and maxValue
function [ TrainPredicted, TestPredicted ] = ...
    Algorithm_TrainAndPredict_Strong(...
    Gtrain,...      % Matrix of user relations in train set
    ...             %
    Ytrain,...      % Matrix of user - artist listen counts in train    
    ...             % 
    Gtrain_test,... % Matrix of user relations between train and test set
    ...             %
    Gtest,...       % Matrix of user relations in test set
    ...             %
    varargin...     %
)
    [P,varargin] = varargGet('Alg_P', varargin);
    [maxValue,varargin] = varargGet('Alg_maxValue', varargin);
    % Possible value range
    Population = 0:maxValue;
    Weights = [P repmat((1 - P) / maxValue, 1, maxValue)];
    setSeed(1);
    N = size(Ytrain, 1);
    M = size(Ytrain, 2);
    K = size(Gtest, 1);
    TrainPredicted = reshape(randsample(Population, N * M, true, Weights), N, M);
    TestPredicted = reshape(randsample(Population, K * M, true, Weights), K, M);
end

