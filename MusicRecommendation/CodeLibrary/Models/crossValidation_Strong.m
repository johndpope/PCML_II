% Function for running cross-validation given a particular
% recommender algorithm for STRONG generalization.
% Here we assume that we try to predict for new users.
%
% This is a very generic function, and it can be used
% for any algorithm. It can also be used when we do CV
% to optimize model parameters or when we do CV to find
% the expected train / test error
%
% inputs:
% Gtrain - matrix of user relations
% Ytrain - matrix of user - artist listen counts
% CallFunction - function to be called as a result of CV
% This function should return as first 2 arguments predictions
% on the train and test set.
%
% varagin - argument list: 'ParameterName1', Value1,
% 'ParameterName2', Value2, etc.
%
% This argument list should contain BOTH arguments
% passed to the CV function and to the function that is called inside.
%
% Naming convention of arguments:
% arguments to be passed to cross-validation function should have name
% 'CV_...'. See examples below.
% 
% Required parameters:
% 'CV_type' - either 'CV' or 'Split'.
% 
% 'CV': K-fold-cross-validation, 
%  requires 'CV_k' in varargin
% 'Split': random train-test split
%  requires 0<'CV_p'<1, 'CV_k'
%
% outputs:
% TrainError - vector of train errors for each fold
% TestError - vector of test errors for each fold
% 
function [ TrainError, TestError ] = crossValidation_Strong( ...
    Gtrain, ...                     % Sparse friendship graph
    ...                             %
    Ytrain, ...                     % Sparse listen count
    ...                             %
    TrainAndPredictFunction, ...    % Function for training model on
    ...                             % the training data and testing
    ...                             % on the test data
    ...                             % Required output is stated in 
    ...                             % Models/ directory
    ...                             %
    varargin...                     % Various additional arguments to 
    ...                             % be passed in either this function or
    ...                             % TrainAndPredict function
    ...                             % in a typical matlab format:
    ...                             % pairs 'Name', value.
    ...                             % Ex. 'CV_k', 3, 'Opt_lambda', [1 2 3]
    )
    
    % First, extract all necessary parameters from varargin
    % It is always the same line:
    % [Parameter, varargin] = varargGet('ParameterName', varargin)
    % ex. below
    [seed,varargin] = varargGet('CV_seed', varargin);
    [CV_type,varargin] = varargGet('CV_type', varargin);
    [verbose,varargin] = varargGet('CV_verbose', varargin);
    
    if (strcmp(CV_type, 'CV')) 
        [K,varargin] = varargGet('CV_k', varargin);
    elseif (strcmp(CV_type, 'Split'))
        [P,varargin] = varargGet('CV_p', varargin);
        [K,varargin] = varargGet('CV_k', varargin);
    end
    
    % Default verbose behaviour: 
    % verbose = 0 -> no output to console from a function
    % verbose > 0 -> output summary to console
    % verbose > 1 -> output results for each iteration of the loop
    if (verbose > 0) 
        fprintf('Starting cross-validation\n');
        fprintf('CV type: %s\n', CV_type);
        if (strcmp(CV_type, 'CV')) 
            fprintf('Number of folds: %d\n', K);
        elseif (strcmp(CV_type, 'Split'))
            fprintf('Number of folds: %d\n', K);
            fprintf('Probability of training set: %d\n', P);
        end
    end

    ne_idx = find(full(sum(Ytrain, 2)) > 0);
    NE = length(ne_idx);
    
    N = size(Ytrain, 1);
    M = size(Ytrain, 2);
    
    setSeed(seed);
    
    % Makes a train_test split based on the parameters passed
    if (strcmp(CV_type, 'CV')) 
        nk = floor(NE / K);
        TrainIdx = zeros(K, NE - nk);
        TestIdx = zeros(K, nk);
        perm = randperm(NE);
        idx = zeros(NE, 1);
        for i = 1:K
            idx(perm(((i - 1) * nk + 1):(i * nk))) = i;
        end
        for i = 1:K
            TrainIdx(i,:) = ne_idx((idx ~= i));
            TestIdx(i,:) = ne_idx((idx == i));
        end
    elseif (strcmp(CV_type, 'Split'))
        trainPart = floor(NE * P);
        TrainIdx = zeros(K, trainPart);
        TestIdx = zeros(K, NE - trainPart);
        for i = 1:K
            perm = randperm(NE);
            TrainIdx(i,:) = ne_idx(perm(1:trainPart));
            TestIdx(i,:) = ne_idx(perm((trainPart+1):end));
        end
    else
        print 'ERROR - wrong CV type';
        return;
    end
    
    % Matrixes
    TrainError = zeros(K, 1);
    TestError = zeros(K, 1);
    % Main execution loop 
    for foldIdx = 1:K
        % Calling TrainAndPredict function, extracting first 2 parameters
        % We pass same varagin that we received here to the function
        TrI = TrainIdx(foldIdx,:);
        TeI = TestIdx(foldIdx,:);
       
        G_tr = sparse(N, N);  G_tr(TrI, TrI) = Gtrain(TrI, TrI);
        G_tr_te = sparse(N, N); G_tr_te(TrI, TeI) = Gtrain(TrI, TeI);
        G_te_tr = sparse(N, N); G_te_tr(TeI, TrI) = Gtrain(TeI, TrI);
        G_te = sparse(N, N); G_te(TeI, TeI) = Gtrain(TeI, TeI);
        Y_tr = sparse(N, M); Y_tr(TrI, :) = Ytrain(TrI,:);
        Y_te = sparse(N, M); Y_te(TeI, :) = Ytrain(TeI,:);
        
        [TrainPredicted, TestPredicted] = ...
            TrainAndPredictFunction(...
            G_tr, Y_tr, G_tr_te, G_te_tr, G_te, Y_te,...
            varargin);
        % Compute RMSE. 
        % !!! - See note in RMSE function
         TrainError(foldIdx) = RMSE(TrainPredicted, Y_tr);
         TestError(foldIdx) = RMSE(TestPredicted, Y_te);
         if (verbose > 1)
            fprintf('Fold  | Train Error  |  Test Error\n');
            fprintf('%03d   |    %0.4f     |    %0.4f\n\n',...
                foldIdx, TrainError(foldIdx), TestError(foldIdx));
        end 
    end
    if (verbose > 0)
        fprintf('Predicted test error  = %0.4f ~ %0.4f SD\n',...
            mean(TestError), std(TestError));
        fprintf('Predicted train error = %0.4f ~ %0.4f SD\n',...
            mean(TrainError), std(TrainError));
    end
end

