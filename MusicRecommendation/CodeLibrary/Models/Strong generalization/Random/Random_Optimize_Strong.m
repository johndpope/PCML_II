% Random prediction
% Used for STRONG generalization - prediction for new users
function [ TrainPredicted, TestPredicted,... 
    best_P,...                       % best selected P value
    ...                             % 
    best_MV,...                      % best selected maxvalue
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
    Gtest,...       % Matrix of user relations in test set
    ...             %
    varargin...     % Additional arguments to be passed
)
    
    [CV_k,varargin] = varargGet('Opt_CV_k', varargin);
    [CV_seed,varargin] = varargGet('Opt_CV_seed', varargin);
    [P_values,varargin] = varargGet('Opt_P_values', varargin);
    [maxValues,varargin] = varargGet('Opt_maxValues', varargin);
    
    TrainError = zeros(length(P_values), length(maxValues));
    TestError = zeros(length(P_values), length(maxValues));
    
    % do CV to find optimal parameter values
     for P_idx = 1 : length(P_values)
         for MV_idx = 1 : length(maxValues)
             [TrainErrors, TestErrors] = crossValidation_Strong(...
                 Gtrain, Ytrain, @Random_TrainAndPredict_Strong,...
                 varargin, 'CV_type', 'CV', 'CV_k',CV_k,...
                 'CV_seed', CV_seed,...
                 'Alg_P', P_values(P_idx), ...
                 'Alg_maxValue', maxValues(MV_idx));
             TrainError(P_idx, MV_idx) = mean(TrainErrors);
             TestError(P_idx, MV_idx) = mean(TestErrors);
         end
     end
     
    [minCVTestError, minIdx] = min(TestError(:));
    [min_P_idx, min_MV_idx] = ind2sub(size(TestError), minIdx);
    best_P = P_values(min_P_idx);
    best_MV = maxValues(min_MV_idx);
    
    expectedTrainError = TrainError(minIdx);
    expectedTestError = TestError(minIdx);
    
    [TrainPredicted, TestPredicted] = Random_TrainAndPredict_Strong(...
        Gtrain, Ytrain, Gtrain_test, Gtest,...
        'Alg_P', best_P, 'Alg_maxValue', best_MV);
    
    % Train Model on the whole dataset 
    % [TrainPredicted, TestPredicted, other staff you want] = ...
    % Algorithm_TrainAndPredict(Gtrain, Ytrain, Gtrain_test, Gtest,
    % params);
end

