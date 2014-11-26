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
    best_P,...                      % best selected P value
    ...                             % 
    best_MV,...                     % best selected maxvalue
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
    [P_values,varargin] = varargGet('Opt_P_values', varargin);
    [maxValues,varargin] = varargGet('Opt_maxValues', varargin);
    [verbose,varargin] = varargGet('Opt_verbose', varargin);
    
    % Default verbose behaviour: 
    % verbose = 0 -> no output to console from a function
    % verbose > 0 -> output summary to console
    % verbose > 1 -> output results for each iteration of the loop
    if (verbose > 0) 
        fprintf('Starting parameter optimization\n');
        fprintf('Number of P values: %d\n', length(P_values));
        fprintf('Number of maxValues: %d\n', length(maxValues));
    end
    
    % Define train and test error matrices. They can be
    % more than 2 dimensional
    TrainError = zeros(length(P_values), length(maxValues));
    TestError = zeros(length(P_values), length(maxValues));
    
    
    if (verbose > 1)
        fprintf('  P    | maxValue | TestError | TrainError  \n');
        fprintf('--------------------------------------------\n');
    end
    
    % Try each combination of parameters, and do cross-validation:
     for P_idx = 1 : length(P_values)
         for MV_idx = 1 : length(maxValues)
             %
             % Call cross-validation that calls 
             % Algorithm_TrainAndPredict_Type function
             % as varargin, submit varargin passed to the current function
             % plus parameter values for the algorithm (ex P, maxValue)
             % plus parameter values for CV (ex CV_seed, CV_verbose)
             [TrainErrors, TestErrors] = crossValidation_Strong(...
                 Gtrain, Ytrain, @Random_TrainAndPredict_Strong,...
                 varargin, 'CV_type', 'CV', 'CV_k',CV_k,...
                 'CV_seed', 1, 'CV_verbose', 0,...
                 'Alg_P', P_values(P_idx), ...
                 'Alg_maxValue', maxValues(MV_idx));
             % Averaging over train / test errors
             TrainError(P_idx, MV_idx) = mean(TrainErrors);
             TestError(P_idx, MV_idx) = mean(TestErrors);
             if (verbose > 1)
                 fprintf('%0.4f |  %02d      |   %0.4f  |   %0.4f\n',...
                     P_values(P_idx), maxValues(MV_idx),...
                     TestError(P_idx, MV_idx), TrainError(P_idx, MV_idx));
             end
         end
     end
     
    % Dimension-independent code for finding minimum value:
    [minCVTestError, minIdx] = min(TestError(:));
    % [min_parameter_1_idx, ..., min_parameter_i_idx, ..] = 
    % ind2sub(size(TestError), minIdx)
    [min_P_idx, min_MV_idx] = ind2sub(size(TestError), minIdx);
    % best_parameter_i_value = ...
    %   parameter_i_values_array(min_parameter_i_idx)
    % For each of the parameters in the optimization
    best_P = P_values(min_P_idx);
    best_MV = maxValues(min_MV_idx);
    
    % Expected train and test error for optimal parameter values
    % expectedSMTHError = SMTHError(...,min_parameter_i_idx,...)
    expectedTrainError = TrainError(min_P_idx, min_MV_idx);
    expectedTestError = TestError(min_P_idx, min_MV_idx);
    
    if (verbose > 0)
        fprintf('Best P               = %0.4f\n', best_P);
        fprintf('Best maxValue        = %02d\n', best_MV);
        fprintf('Expected TestError   = %0.4f\n', expectedTestError);
        fprintf('Expected TrainError  = %0.4f\n', expectedTrainError);
    end
    
    % Call train and predict to predict on the actual train / test set
    % Use optimal parameter values found above
    % Generally, as a good practice, algorithm should only need these
    % parameters optimized above. If algorithm requires some other hyper
    % parameters, that are passed through the varargin but not defined 
    % here, probably something is wrong.
    [TrainPredicted, TestPredicted] = Random_TrainAndPredict_Strong(...
        Gtrain, Ytrain, Gtrain_test, Gtest_train, Gtest,...
        'Alg_P', best_P, 'Alg_maxValue', best_MV);
end

