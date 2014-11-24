% Random algorithm
% Generates 0 with probability p
% or number between 1 and maxValue uniformly
% Used for STRONG generalization - predict for new users
%
% EXAMPLE FOR OTHER ALGORITHM IMPLEMENTATION
%
% General description of Algorithm_TrainAndPredict_Strong
%
% inputs:
% Gtrain - matrix of user relations in train
% Ytrain - user - artist matrix in train
% Gtrain_test - matrix of user relations in train and test
% Gtest - matrix of user relations in test
% varagin - argument list: 'ParameterName1', Value1,
% 'ParameterName2', Value2, etc.
%
% Naming convention of arguments:
% arguments to be passed to TrainAndPredict function should have name
% 'Alg_...'. See examples below.
%
% outputs:
% TrainPredicted - predictions on the train set
% TestPredicted - predictions on the test set
% Possibly - model for more complex algorithms
%
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

