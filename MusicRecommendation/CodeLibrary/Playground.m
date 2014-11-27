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

% % Relatively interesting dense subset of users and artists
% UsersTr = [2 3 9 11]; UsersTe = [29];
% Artists = [153 201 284 980 1079];
% 
% Somewhat bigger option
 UsersTr = 1:500; UsersTe = 501:700;
 Artists = 1:size(Ytrain, 2);
%
% Whole dataset
% UsersTr = 1:1500; UsersTe = 1501:size(Gtrain, 1);
% % Artists = 1:size(Ytrain, 2);

% Always create matrixes of full size
N = size(Ytrain, 1);
M = size(Ytrain, 2);

G_tr = sparse(N, N);
G_tr_te = sparse(N, N);
G_te_tr = sparse(N, N);
G_te = sparse(N, N);
Y_tr = sparse(N, M);
Y_te = sparse(N, M);

% Fill in only necessary info
G_tr(UsersTr, UsersTr) = Gtrain(UsersTr, UsersTr);
G_tr_te(UsersTr, UsersTe) = Gtrain(UsersTr, UsersTe);
G_te_tr(UsersTe, UsersTr) = Gtrain(UsersTe, UsersTr);
G_te(UsersTe, UsersTe) = Gtrain(UsersTe, UsersTe);
Y_tr(UsersTr, Artists) = Ytrain(UsersTr, Artists);
Y_te(UsersTe, Artists) = Ytrain(UsersTe, Artists);

G = G_tr + G_tr_te + G_te_tr + G_te;
Y = Y_tr + Y_te;

% Visualize sparcity patterns in train
figure;
a = subplot(2, 1, 1);
spy(G_tr(UsersTr, UsersTr));
b = subplot(2, 1, 2);
spy(Y_tr(UsersTr, Artists));

%% Simple algorithm call example

% hyperparameters
p = 0.1;
maxValue = 16;


[TrainPredicted, TestPredicted] = Random_TrainAndPredict_Strong(...
    ... % Train and test parameters
    G_tr, Y_tr, G_tr_te, G_te_tr, G_te, Y_te,...
    ... % algorithm hyperparameters
    'Alg_P', p, 'Alg_maxValue', maxValue);

%% Training the model and testing on the testset
% With hyperparameter optimization

% Optimizing model parameters
% P_values = 0:0.1:1;
P_values = 0;
maxValues = 800:20:1500;
[TrainPredicted, TestPredicted, best_P, best_MV, ...
    expectedTrainError, expectedTestError,...
    TrainError, TestError] = Random_Optimize_Strong(...
    ... % Train and test parameters
    G_tr, Y_tr, G_tr_te, G_te_tr, G_te, Y_te,...
    ... % Hyperparameters to be optimized
    'Opt_P_values', P_values, 'Opt_maxValues', maxValues,...
    ... % Hyperparameters passed to CV from Optimize
    'Opt_CV_k', 10, 'Opt_CV_seed', 1,...
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
maxValues = 950:10:1050;
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

%% Learning curve for PerArtist
P = 0.1:0.1:0.9;
TrE = zeros(length(P), 1);
TeE = zeros(length(P), 1);
for idx=1:length(P)
    fprintf('%d\n', P(idx));
    [TrainError, TestError] = crossValidation_Strong(...
        ... % Only training data is provided
        G, Y,...
        ... % Function to be called
        @ConstantPerArtist_TrainAndPredict_Strong,...
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
