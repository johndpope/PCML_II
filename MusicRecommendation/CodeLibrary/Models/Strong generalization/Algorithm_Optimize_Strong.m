% Template function to be copied to implement other models
% Used for STRONG generalization - prediction for new users
function [ TrainPredicted, TestPredicted... 
    ...                             %, additional output you want
    ...                             % i.e. best parameter values
    ...                             % model trained on whole dataset, etc.
    ] = Algorithm_Optimize_Strong(... 
    Gtrain,...      % Matrix of user relations in train set
    ...             %
    Ytrain,...      % Matrix of user - artist listen counts in train    
    ...             % 
    Gtrain_test,... % Matrix of user relations between train and test set
    ...             %
    Gtest,...       % Matrix of user relations in test set
    ...             %
    varargin...     % Additional arguments to be passed
)

    
    % first, parse varargin
    %[CV_type,varargin] = varargGet('Opt_CV_type', varargin);
    %[lambdas,varargin] = varargGet('Opt_lambda', varargin);
    
    % do CV to find optimal parameter values
    % for lambda = 1 : 10
    % for beta = 1 : 10
    % 
    %
    % [TrainErrors, TestErrors] = crossValidation(Gtrain, Ytrain,
    % Algo_CV_type, Algorithm_TrainAndPredict, 
    % varargin, 'CV_type', 'CV', 'CV_K', 5, 
    % 'lambda', lambda, .. other params you want)
    % TrainError(lambda, beta) = mean(TrainErrors);
    % TestError(lambda, beta) = mean(TestErrors);
    % endfor
    % endfor
    %
    % find best lambda, beta - minimum value in TestError
    %
    
    % Train Model on the whole dataset 
    % [TrainPredicted, TestPredicted, other staff you want] = ...
    % Algorithm_TrainAndPredict(Gtrain, Ytrain, Gtrain_test, Gtest,
    % params);
end

