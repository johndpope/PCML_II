% Random prediction
% Used for STRONG generalization - prediction for new users
%
% EXAMPLE FOR OTHER ALGORITHM IMPLEMENTATION
% 
% General description of Algorithm_Optimize_Strong function:
% 
% inputs: Gtrain, Ytrain, Gtrain_test, Gtest, varargin
% Gtrain - matrix of user relations in train
% Ytrain - user - artist matrix in train
% Gtrain_test - matrix of user relations in train and test
% Gtest - matrix of user relations in test
% varagin - argument list: 'ParameterName1', Value1,
% 'ParameterName2', Value2, etc.
%
% Naming convention of arguments:
% arguments to be passed to Optimize function should have name
% 'Opt_...'. See examples below.
%
% outputs: 
% best_value_of_optimization_parameter_1,
% best_value_of_optimization_parameter_2,
% ...,
% expectedTrainError for best parameter values,
% expectedTestError for best parameter values,
%
% TrainError - matrix of train errors for different parameter combinations
% TestError - matrix of test errors for different parameter combinations
%
function [ TrainPredicted, TestPredicted,... 
    best_lambda,...                 % best selected P value
    ...                             % 
    best_K,...                      % best selected maxvalue
    ...                             % 
    expectedTrainError,...          % expected train error with best vals
    ...                             %
    expectedTestError,...           % expected test error with best vals
    ...                             %
    TrainError,...                  % matrix of trainErrors from CV
    ...                             % 
    TestError...                    % matrix of testErrors from CV
    ] = Random_Optimize_Strong(... 
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
    Ytest,...       % Indices, for which we are insterested in answer
    ...             %
    varargin...     % Additional arguments to be passed
)
    % First, extract all necessary parameters from varargin
    % It is always the same line:
    % [Parameter, varargin] = varargGet('ParameterName', varargin)
    % ex. below
    [CV_k,varargin] = varargGet('Opt_CV_k', varargin);
    [CV_l,varargin] = varargGet('Opt_CV_l', varargin);
    [K_values,varargin] = varargGet('Opt_K_values', varargin);
    [lambdas,varargin] = varargGet('Opt_lambdas', varargin);
    [verbose,varargin] = varargGet('Opt_verbose', varargin);
    iters = 20;
    
    % Default verbose behaviour: 
    % verbose = 0 -> no output to console from a function
    % verbose > 0 -> output summary to console
    % verbose > 1 -> output results for each iteration of the loop
    if (verbose > 0) 
        fprintf('Starting parameter optimization\n');
        fprintf('Number of K values: %d\n', length(K_values));
        fprintf('Number of lambdas: %d\n', length(lambdas));
    end
    
    % Define train and test error matrices. They can be
    % more than 2 dimensional
    TrainError = zeros(length(K_values), length(lambdas));
    TestError = zeros(length(K_values), length(lambdas));
    
    
    
    % Try each combination of parameters, and do cross-validation:
     for K_idx = 1 : length(K_values)
         for L_idx = 1 : length(lambdas)
             %
             % Call cross-validation that calls 
             % Algorithm_TrainAndPredict_Type function
             % as varargin, submit varargin passed to the current function
             % plus parameter values for the algorithm (ex P, maxValue)
             % plus parameter values for CV (ex CV_seed, CV_verbose)
             [TrainErrors, TestErrors] = crossValidation_Weak(...
                 Gtrain, Ytrain, @ALS_TrainAndPredict,...
                 varargin, 'CV_k',CV_k, 'CV_l', CV_l,...
                 'CV_seed', 1, 'CV_verbose', 0,...
                 'Alg_numOfIterations', iters, ...
                 'Alg_numOfFeatures', K_values(K_idx),...
                 'Alg_lambda', lambdas(L_idx));
             % Averaging over train / test errors
             TrainError(K_idx, L_idx) = mean(TrainErrors);
             TestError(K_idx, L_idx) = mean(TestErrors);
             if (verbose > 1)
                 fprintf('%0.4f |  %0.4f      |   %0.4f  |   %0.4f\n',...
                     K_values(K_idx), lambdas(L_idx),...
                     TestError(K_idx, L_idx), TrainError(K_idx, L_idx));
             end
         end
     end
     
    % Dimension-independent code for finding minimum value:
    [minCVTestError, minIdx] = min(TestError(:));
    % [min_parameter_1_idx, ..., min_parameter_i_idx, ..] = 
    % ind2sub(size(TestError), minIdx)
    [min_K_idx, min_L_idx] = ind2sub(size(TestError), minIdx);
    % best_parameter_i_value = ...
    %   parameter_i_values_array(min_parameter_i_idx)
    % For each of the parameters in the optimization
    best_K = K_values(min_K_idx);
    best_lambda = lambdas(min_L_idx);
    
    % Expected train and test error for optimal parameter values
    % expectedSMTHError = SMTHError(...,min_parameter_i_idx,...)
    expectedTrainError = TrainError(min_K_idx, min_L_idx);
    expectedTestError = TestError(min_K_idx, min_L_idx);
    
    if (verbose > 0)
        fprintf('Best K               = %0.4f\n', best_K);
        fprintf('Best lambda          = %0.4f\n', best_L);
        fprintf('Expected TestError   = %0.4f\n', expectedTestError);
        fprintf('Expected TrainError  = %0.4f\n', expectedTrainError);
    end
    
    % Call train and predict to predict on the actual train / test set
    % Use optimal parameter values found above
    % Generally, as a good practice, algorithm should only need these
    % parameters optimized above. If algorithm requires some other hyper
    % parameters, that are passed through the varargin but not defined 
    % here, probably something is wrong.
    [TrainPredicted, TestPredicted] = ALS_TrainAndPredict(...
        Gtrain, Ytrain, Gtrain_test, Gtest_train, Gtest, Ytest,...
        'Alg_numOfFeatures', best_K,...
        'Alg_lambda', best_lambda,...
        'Alg_numOfIterations', iters);
end

