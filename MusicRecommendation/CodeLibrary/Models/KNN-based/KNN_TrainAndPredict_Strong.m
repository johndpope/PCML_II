function [ TrainPredicted, TestPredicted ] = ...
    KNN_TrainAndPredict_Strong(...
    Gtrain,...      % Matrix of user relations in train set
    ...             %
    Ytrain,...      % Matrix of user - artist listen counts in train    
    ...             % 
    Gtrain_test,... % Matrix of user relations between train and test set
    ...             %
    Gtest_train,... % Matrix of user relations between test and train set
    ...             %
    Gtest,...       % Matrix of user relations in test set
    ...             %
    varargin...     %
)
    % First, extract all necessary parameters from varargin
    % It is always the same line:
    % [Parameter, varargin] = varargGet('ParameterName', varargin)
    % ex. below
    [P,varargin] = varargGet('Alg_P', varargin);
    [maxValue,varargin] = varargGet('Alg_maxValue', varargin);
    
    % Actually create a model on the train set
    % For this algorithm model is independent of the train set
    % Define possible output values
    Population = 0:maxValue;
    % Define weights - P for 0, uniformly distributed for all other values
    Weights = [P repmat((1 - P) / maxValue, 1, maxValue)];
    setSeed(1);
    N = size(Ytrain, 1);
    M = size(Ytrain, 2);
    K = size(Gtest, 1);
    
    % Use the trained model on the train and test set
    % Output should be matrixes
    % For normal algorithms they probably should be sparse
    % Matlab seamlessly operates with sparse and normal matrixes
    % So that doesn't make a difference
    TrainPredicted = reshape(...
        randsample(Population, N * M, true, Weights), N, M);
    TestPredicted = reshape(...
        randsample(Population, K * M, true, Weights), K, M);
end

