%% Initial cleaning
clear all;
clc;
%% Data analysis and cleaning
load 'Data/songTrain.mat'
Gtrain_Copy = Gtrain;
Ytrain_Copy = Ytrain;

%% Example of application of one method to the data

% Relatively interesting dense subset of users and artists
Users = [2 3 7 9 11];
Artists = [153 201 284 980 1079];

G = Gtrain_Copy(Users, Users);
Y = Ytrain_Copy(Users, Artists);
% Make smaller values for the sake of example
Y = mod(Y, 17);

% Plot sparcity patterns
figure;
subplot(1, 2, 1);
spy(G);
subplot(1, 2, 2);
spy(Y);

% Define train and test data
Gtrain = G(1:4,1:4);
Gtrain_test = G(1:4,5);
Gtest = G(5,5);
Ytrain = Y(1:4,:);
Ytest = Y(5,:);

% Call simple model
% probability of generating 0
p = 0.1;
% maximum possible generated value
maxValue = 16;

% Simple model call
[TrainPredicted, TestPredicted] = Random_TrainAndPredict_Strong(...
    Gtrain, Ytrain, Gtrain_test, Gtest, 'Alg_P', p, 'Alg_maxValue', maxValue);

% Optimizing model parameters
P_values = 0:0.1:1;
maxValues = 2:20;
[TrainPredicted, TestPredicted, best_P, best_MV, ...
    expectedTrainError, expectedTestError,...
    TrainError, TestError] = Random_Optimize_Strong(Gtrain, Ytrain,...
    Gtrain_test, Gtest, 'Opt_P_values', P_values, ...
    'Opt_maxValues', maxValues, 'Opt_CV_k', 4, 'Opt_CV_seed', 1);

% Cross validating on the optimal algorithm to get truer estimates
P_values = 0:0.1:1;
maxValues = 2:20;
[TrainError, TestError] = crossValidation_Strong(G, Y,...
    @Random_Optimize_Strong,...
    'CV_type', 'CV', 'CV_k', 5,'CV_seed', 1,...
    'Opt_P_values', P_values, 'Opt_maxValues', maxValues,...
    'Opt_CV_k', 4, 'Opt_CV_seed', 1);

% Now doing the same for bigger data
G = Gtrain_Copy;
Y = Ytrain_Copy;
Y = mod(Y, 17);
figure;
subplot(1, 2, 1);
spy(G);
subplot(1, 2, 2);
spy(Y);
Gtrain = G(1:1500, 1:1500);
Gtrain_test = G(1:1500, 1501:end);
Gtest = G(1501:end, 1501:end);
Ytrain = Y(1:1500,:);
Ytest = Y(1501:end,:);

% Optimizing model parameters
P_values = 0;
maxValues = 10:20;
[TrainPredicted, TestPredicted, best_P, best_MV, ...
    expectedTrainError, expectedTestError,...
    TrainError, TestError] = Random_Optimize_Strong(Gtrain, Ytrain,...
    Gtrain_test, Gtest, 'Opt_P_values', P_values, ...
    'Opt_maxValues', maxValues, 'Opt_CV_k', 10, 'Opt_CV_seed', 1);


% Cross validating on the optimal algorithm to get truer estimates
tic;
P_values = 0:0.1:1;
maxValues = 2:20;
[TrainError, TestError] = crossValidation_Strong(G, Y,...
    @Random_Optimize_Strong,...
    'CV_type', 'CV', 'CV_k', 10,'CV_seed', 1,...
    'Opt_P_values', P_values, 'Opt_maxValues', maxValues,...
    'Opt_CV_k', 10, 'Opt_CV_seed', 1);
toc;




% First - let's simply call the model