function [ TrainPredicted, TestPredicted,... 
    best_split,...                      % best selected P value
    ...                             % 
    expectedTrainError,...          % expected train error with best vals
    ...                             %
    expectedTestError,...           % expected test error with best vals
    ...                             %
    TrainError,...                  % matrix of trainErrors from CV
    ...                             % 
    TestError...                    % matrix of testErrors from CV
    ] = KNN_Java_Optimize(... 
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
    [Ns,varargin] = varargGet('Opt_splits', varargin);
    [verbose,varargin] = varargGet('Opt_verbose', varargin);
    
    % Default verbose behaviour: 
    % verbose = 0 -> no output to console from a function
    % verbose > 0 -> output summary to console
    % verbose > 1 -> output results for each iteration of the loop
    if (verbose > 0) 
        fprintf('Starting parameter optimization\n');
        fprintf('Number of split values: %d\n', length(Ns));
    end
    
    % Define train and test error matrices. They can be
    % more than 2 dimensional
    TrainError = zeros(length(Ns), 1);
    TestError = zeros(length(Ns), 1);
    
    
    % Try each combination of parameters, and do cross-validation:
     for N_idx = 1 : length(Ns)
         [TrainErrors, TestErrors] = crossValidation_Weak(...
             Gtrain, Ytrain, @KNN_Mix,...
             varargin, 'CV_k',CV_k, 'CV_l', CV_l,...
             'CV_seed', 1, 'CV_verbose', 0,...
             'Alg_split', Ns(N_idx));
         % Averaging over train / test errors
         TrainError(N_idx) = mean(TrainErrors);
         TestError(N_idx) = mean(TestErrors);
         if (verbose == 2) 
            fprintf('N = %03d Tr = %0.4f Te = %0.4f\n', ...
                Ns(N_idx), TrainError(N_idx), TestError(N_idx));
         end

     end
     
    % Dimension-independent code for finding minimum value:
    [minCVTestError, minIdx] = min(TestError(:));
    % [min_parameter_1_idx, ..., min_parameter_i_idx, ..] = 
    % ind2sub(size(TestError), minIdx)
    [min_N_idx] = ind2sub(size(TestError), minIdx);
    % best_parameter_i_value = ...
    %   parameter_i_values_array(min_parameter_i_idx)
    % For each of the parameters in the optimization
    best_split = Ns(min_N_idx);
    
    % Expected train and test error for optimal parameter values
    % expectedSMTHError = SMTHError(...,min_parameter_i_idx,...)
    expectedTrainError = TrainError(min_N_idx);
    expectedTestError = TestError(min_N_idx);
    
    if (verbose > 0)
        fprintf('Best Split           = %0.4f\n', best_split);
        fprintf('Expected TestError   = %0.4f\n', expectedTestError);
        fprintf('Expected TrainError  = %0.4f\n', expectedTrainError);
    end
    
    % Call train and predict to predict on the actual train / test set
    % Use optimal parameter values found above
    % Generally, as a good practice, algorithm should only need these
    % parameters optimized above. If algorithm requires some other hyper
    % parameters, that are passed through the varargin but not defined 
    % here, probably something is wrong.
    [TrainPredicted, TestPredicted] = KNN_Mix(...
        Gtrain, Ytrain, Gtrain_test, Gtest_train, Gtest, Ytest,...
        'Alg_split', best_split);
end

