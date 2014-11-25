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

    N = size(Ytrain, 1); % # of users
    setSeed(seed);
    
    % Makes a train_test split based on the parameters passed
    if (strcmp(CV_type, 'CV')) 
        TrainIdx = false(K, N);
        TestIdx = false(K, N);
        perm = randperm(N);
        idx = zeros(N, 1);
        nk = floor(N / K);
        for i = 1:K
            idx(perm(((i - 1) * nk + 1):(i * nk))) = i;
        end
        for i = 1:K
            TrainIdx(i,:) = (idx ~= i);
            TestIdx(i,:) = (idx == i);
        end
    elseif (strcmp(CV_type, 'Split'))
        trainPart = floor(N * P);
        TrainIdx = zeros(K, trainPart);
        TestIdx = zeros(K, N - trainPart);
        for i = 1:K
            perm = randperm(N);
            TrainIdx(i,:) = perm(1:trainPart);
            TestIdx(i,:) = perm((trainPart+1):end);
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
        [TrainPredicted, TestPredicted] = ...
            TrainAndPredictFunction(...
            Gtrain(TrainIdx(foldIdx,:), TrainIdx(foldIdx,:)),...
            Ytrain(TrainIdx(foldIdx,:),:),...
            Gtrain(TrainIdx(foldIdx,:), TestIdx(foldIdx,:)),...
            Gtrain(TestIdx(foldIdx,:), TrainIdx(foldIdx,:)),...
            Gtrain(TestIdx(foldIdx,:), TestIdx(foldIdx,:)),...
            varargin);
        % Compute RMSE. 
        % !!! - See note in RMSE function
         TrainError(foldIdx) = RMSE(TrainPredicted, ...
                                    Ytrain(TrainIdx(foldIdx,:),:));
         TestError(foldIdx) = RMSE(TestPredicted, ...
                                   Ytrain(TestIdx(foldIdx,:),:));
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

