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
    stackAlgorithm(...
    Ytrain,...      % Matrix of user - artist listen counts in train    
    Ytest,...       % Indices, for which we are insterested in answer
    I_tr, I_te)

 % Transform the data
 % Subtract baseline
 % Run algorithm
 % Add baseline
 % Detransform predictions
  
 Y_tr_0 = Ytrain;
 Y_te_0 = Ytest;
 
 % Transform the data
% [muU, stdU, Y_tr_1] = transformData(Y_tr_0, I_tr, Y_tr_0, I_tr);
% [muU, stdU, Y_te_1] = transformData(Y_te_0, I_te, Y_tr_0, I_tr);

% [lambda, muU, stdU, Y_tr_1] = transformData2(Y_tr_0, I_tr, Y_tr_0, I_tr);
% [lambda, muU, stdU, Y_te_1] = transformData2(Y_te_0, I_te, Y_tr_0, I_tr);


 % Subtract baseline
  [meanAll, meanU, meanI, Y_tr_2] = subtractBaseline(Y_tr_0, I_tr, Y_tr_0, I_tr);
  [meanAll, meanU, meanI, Y_te_2] = subtractBaseline(Y_te_0, I_te, Y_tr_0, I_tr);
 
 % Run algorithm
 % P_tr_2 = Y_tr_2; %sparse(size(Y_tr_2, 1), size(Y_tr_2, 2));
 % P_te_2 = Y_te_2; %sparse(size(Y_te_2, 1), size(Y_te_2, 2));
 
  P_tr_2 = sparse(size(Y_tr_2, 1), size(Y_tr_2, 2));
  P_te_2 = sparse(size(Y_te_2, 1), size(Y_te_2, 2));
% [P_tr_2, P_te_2] = ALS_TrainAndPredict(Gtrain, Y_tr_1, Gtrain_test, Gtest_train, Gtest, Y_te_1,...
%     'Alg_numOfIterations', 20,...
%     'Alg_numOfFeatures', 10,...
%     'Alg_lambda', 0.0001);
 
%  figure;
%  
%  subplot(1,2,1); hist(Y_tr_1(I_tr), 100);
%  subplot(1,2,2); hist(Y_tr_2(I_tr), 100);
%  %subplot(1,3,1); hist(Y_tr_0(I_tr));
%  %subplot(1,3,2); hist(Y_tr_1(I_tr), 100);
%  %subplot(1,3,3); 
%  
%  
%  figure;
%  % subplot(1,3,1); hist(Y_te_0(I_te));
%  subplot(1,2,1); hist(Y_te_1(I_te), 100);
%  subplot(1,2,2); hist(Y_te_2(I_te), 100);
%   
 % Add baseline
 P_tr_0 = addBaseline(meanAll, meanU, meanI, P_tr_2, I_tr);
 P_te_0 = addBaseline(meanAll, meanU, meanI, P_te_2, I_te);
 
  % Detransform the data
 % P_tr_0 = DEtransformData(muU, stdU, P_tr_1, I_tr, Y_tr_0, I_tr);
 % P_te_0 = DEtransformData(muU, stdU, P_te_1, I_te, Y_tr_0, I_tr);
 
% P_tr_0 = DEtransformData2(lambda, muU, stdU, P_tr_1, I_tr, Y_tr_0, I_tr);
% P_te_0 = DEtransformData2(lambda, muU, stdU, P_te_1, I_te, Y_tr_0, I_tr);
 
 
 % Return results
 TrainPredicted = P_tr_0;
 TestPredicted = P_te_0;
 
end

