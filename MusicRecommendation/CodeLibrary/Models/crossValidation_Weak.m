% Function for running cross-validation given a particular
% recommender algorithm for WEAK generalization.
% Here we assume that we try to predict for unknown artists;
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
%  requires 'CV_k' - number of folds
%  requiers 'CV_l' - number of ratings to go to test
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
    [verbose,varargin] = varargGet('CV_verbose', varargin);
    
    [L,varargin] = varargGet('CV_l', varargin);
    [K,varargin] = varargGet('CV_k', varargin);
    
    % Default verbose behaviour: 
    % verbose = 0 -> no output to console from a function
    % verbose > 0 -> output summary to console
    % verbose > 1 -> output results for each iteration of the loop
    if (verbose > 0) 
        fprintf('Starting cross-validation\n');
        fprintf('Number of folds: %d\n', K);
        fprintf('Number of ratings going to test: %d\n', L);
    end

    N = size(Ytrain, 1);
    M = size(Ytrain, 2);
    
    u_Sums = full(sum(boolean(Ytrain), 2));
    
    % Matrixes
    TrainError = zeros(K, 1);
    TestError = zeros(K, 1);
    % Main execution loop 
    for foldIdx = 1:K
        % Calling TrainAndPredict function, extracting first 2 parameters
        % We pass same varagin that we received here to the function
        
        G_tr = Gtrain; 
        %All Gs are going to be equal in this case
        %and include all users
        Y_tr = sparse(N, M); 
        Y_te = sparse(N, M);  
        
        setSeed(seed + foldIdx);
        
        for u=1:N
            if (u_Sums(u) > L) 
                all_idx = find(Ytrain(u,:) > 0);
                all_idx = all_idx(randperm(length(all_idx)));
                Y_tr(u, all_idx(1:(length(all_idx)-L))) =...
              Ytrain(u, all_idx(1:(length(all_idx)-L)));
                Y_te(u, all_idx((length(all_idx)-L+1):end)) =...
              Ytrain(u, all_idx((length(all_idx)-L+1):end));
            else
                Y_tr(u,:) = Ytrain(u,:);
            end
        end
        
        [TrainPredicted, TestPredicted] = ...
            TrainAndPredictFunction(...
            G_tr, Y_tr, G_tr, G_tr, G_tr, Y_te,...
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

