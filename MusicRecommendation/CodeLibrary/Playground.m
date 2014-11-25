%% Description
% 
% This is an example of a script that shows 
% the use of the infrastructure to train model,
% make predictions, estimate errors.
% 
% This is an example for STRONG recommendation
% (for a set of new users)
% 
% Here goes a description of the infrastructure:
% 
% There are 3 functions required to use an algorithm:
% 1). crossValidation_Strong (already implemented)
% 
% A general crossValidation function that is used both when we
% want to do CV to find error estimates and when we try to find
% the optimal value of parameters.
% 
% 2). AlgorithmName_TrainAndPredict_Strong
% 
% Main logic of the algorithm. Input is the train and test data,
% plus a set of hyperparameters for the model (e.x. lambda for LR, K for KNN)
% 
% 3). AlgorithmName_Optimize_Strong
% 
% Function that optimizes hyperparameters of the algorithm. If we want to
% find the predictions on some test set, about which we know nothing, we call
% this function, as it optimizes them.
% Input - train and test data, vectors of parameters to be optimized.
% This function calls the crossValidation_strong function, which itself
% calls the main logic. Thus, we do cross-validation for each parameter 
% combination to estimate the test error.
% 
% There are 2 typical usages of the infrastructure:
% 
% 1). Predict results, given training and testing set
% [TrainPredicted, TestPredicted] = 
%     AlgorithmName_Optimize_Strong(TrainData, TestData, 
%     hyperparameter_vectors_to_be_optimized, other_parameters);
% 
% 2). Estimate training / testing error, given only the training set
% [TrainError, TestError] = crossValidation_Strong(
%     TrainData, parameters_of_CV, hyperparameter_vectors_to_be_optimized,
%     other_parameters).
%     
% Parameters:
% 
% Parameters are defined in a typical matlab nomenclature, 
% 'ParameterName1', Value1, 'ParameterName2', Value2, etc.
% 
% Naming convention for parameters:
% parameters for CV function have names CV_...
% parameters for Optimize function have names Opt_...
% parameters for TrainAndPredict (main algorithm logic) have names Alg_...
% 
% Code below shows typical usage.
%
%
% The example is totally useless random algorithms
% that has 2 hyperparameters:
% p - probability that user - artist listen count is 0
% maxValue - maximum possible user - item listener count
% if listen count is not 0, it is uniformly selected from 1:maxValue
   
    

%% Data analysis and cleaning

clear all;
clc;
load 'Data/songTrain.mat'

Gtrain_Copy = Gtrain;
Ytrain_Copy = Ytrain;

%% dataset selection

% Option 1 - really small: 5 users, 5 items

% % Relatively interesting dense subset of users and artists
% Users = [2 3 7 9 11];
% Artists = [153 201 284 980 1079];
% 
% G = Gtrain_Copy(Users, Users);
% Y = Ytrain_Copy(Users, Artists);
% % Define train and test data
% Gtrain = G(1:4,1:4);
% Gtrain_test = G(1:4,5);
% Gtest = G(5,5);
% Ytrain = Y(1:4,:);
% Ytest = Y(5,:);

% Option 2 - somewhat bigger - 1200 users, all items
G = Gtrain_Copy(1:1200,1:1200);
Y = Ytrain_Copy(1:1200,:);
% Define train and test data
Gtrain = G(1:1000, 1:1000);
Gtrain_test = G(1:1000, 1001:end);
Gtest = G(1001:end, 1001:end);
Ytrain = Y(1:1000,:);
Ytest = Y(1001:end,:);


% Option 3 - whole dataset, although it takes about 30-40 minutes to run
% G = Gtrain_Copy;
% Y = Ytrain_Copy;
% % Define train and test data
% Gtrain = G(1:1500, 1:1500);
% Gtrain_test = G(1:1500, 1501:end);
% Gtest = G(1501:end, 1501:end);
% Ytrain = Y(1:1500,:);
% Ytest = Y(1501:end,:);


% ---------------------------------------------------------
% additionally 
% Make smaller listener counts - for the sake of simplicity

% Y = mod(Y, 17);
% Ytrain = mod(Ytrain, 17);
% Ytest = mod(Ytest, 17);

% Visualize sparcity patterns
figure;
a = subplot(2, 1, 1);
spy(G);
b = subplot(2, 1, 2);
spy(Y);

%% Simple algorithm call example

% hyperparameters
p = 0.1;
maxValue = 16;


[TrainPredicted, TestPredicted] = Random_TrainAndPredict_Strong(...
    ... % Train and test parameters
    Gtrain, Ytrain, Gtrain_test, Gtest,...
    ... % algorithm hyperparameters
    'Alg_P', p, 'Alg_maxValue', maxValue);

%% Training the model and testing on the testset
% With hyperparameter optimization

% Optimizing model parameters
% P_values = 0:0.1:1;
P_values = 0;
maxValues = 800:20:1000;
[TrainPredicted, TestPredicted, best_P, best_MV, ...
    expectedTrainError, expectedTestError,...
    TrainError, TestError] = Random_Optimize_Strong(...
    ... % Train and test parameters
    Gtrain, Ytrain,Gtrain_test, Gtest, ...,
    ... % Hyperparameters to be optimized
    'Opt_P_values', P_values, 'Opt_maxValues', maxValues,...
    ... % Hyperparameters passed to CV from Optimize
    'Opt_CV_k', 4, 'Opt_CV_seed', 1,...
    ... % Other parameters to Opt
    'Opt_verbose', 2);

% Just looking at the behaviuour for different p
figure;
for P_idx = 1:length(P_values)
    hold on;
    plot(maxValues, TestError(P_idx,:));
end

%% Cross-validating the whole algorithm
% Here for the sake of speed for larger datasets
P_values = 0;
maxValues = 10:20;
[TrainError, TestError] = crossValidation_Strong(...
    ... % Only training data is provided
    G, Y,...
    ... % Function to be called
    @Random_Optimize_Strong,...
    ... % CV parameters
    'CV_type', 'CV', 'CV_k', 5,'CV_seed', 1, 'CV_verbose', 2,...
    ... % Hyperparameters to optimize
    'Opt_P_values', P_values, 'Opt_maxValues', maxValues,...
    ... % parameters to be used for CV inside of optimize
    ... % They are parsed in Optimize and added to the parameters
    ... % Passed to the CV by different name (with CV_ prefix)
    ... % Purpose of such approach: avoid naming conflict
    'Opt_CV_k', 4, 'Opt_CV_seed', 1,...
    ... % Other parameters to opt
    'Opt_verbose', 1);

%% Learning curve
% TODO: 
% 1. Check that same optimal point holds true for other algs
% 2. Check with bigger dataset
% QUESTION:
% Different algorithms have different learning curves
% See for example Constant and ConstantPerArtist
% What should be the optimal size?
P = 0.1:0.1:0.9;
TrE = zeros(length(P), 1);
TeE = zeros(length(P), 1);
for idx=1:length(P)
    fprintf('%d\n', P(idx));
    [TrainError, TestError] = crossValidation_Strong(...
        ... % Only training data is provided
        G, Y,...
        ... % Function to be called
        @Constant_TrainAndPredict_Strong,...
        ... % CV parameters
        'CV_type', 'Split', 'CV_k', 200, 'CV_p', P(idx),...
        'CV_seed', 1, 'CV_verbose', 0);
    TrE(idx) = mean(TrainError);
    TeE(idx) = mean(TestError);
end

figure;
plot(P, TrE, '.b');
hold on;
line(P, TrE);
hold on;
plot(P, TeE, '.r');
hold on;
line(P, TeE);


%% Cross-validating random with near-optimal parameter maxValue 900
[TrainError, TestError] = crossValidation_Strong(...
    ... % Only training data is provided
    G, Y,...
    ... % Function to be called
    @Random_TrainAndPredict_Strong,...
    ... % CV parameters
    'CV_type', 'Split', 'CV_k', 100, 'CV_p', 0.7,...
    'CV_seed', 1, 'CV_verbose', 2,...
    'Alg_P', 0, 'Alg_maxValue', 900);

%% Cross-validating the constant algorithm
[TrainError, TestError] = crossValidation_Strong(...
    ... % Only training data is provided
    G, Y,...
    ... % Function to be called
    @Constant_TrainAndPredict_Strong,...
    ... % CV parameters
    'CV_type', 'Split', 'CV_k', 100, 'CV_p', 0.7,...
    'CV_seed', 1, 'CV_verbose', 2);

%% Cross-validating the constantPerArtist algorithm
[TrainError, TestError] = crossValidation_Strong(...
    ... % Only training data is provided
    G, Y,...
    ... % Function to be called
    @ConstantPerArtist_TrainAndPredict_Strong,...
    ... % CV parameters
    'CV_type', 'Split', 'CV_k', 100, 'CV_p', 0.7,...
    'CV_seed', 1, 'CV_verbose', 2);
