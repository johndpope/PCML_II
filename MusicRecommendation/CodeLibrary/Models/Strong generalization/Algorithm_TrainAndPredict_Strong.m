% Template function to be copied to implement other algorithms
% Used for STRONG generalization - predict for new users
function [ TrainPredicted, TestPredicted, Model ] = ...
    Algorithm_TrainAndPredict_Strong(...
    Gtrain,...      % Matrix of user relations in train set
    ...             %
    Ytrain,...      % Matrix of user - artist listen counts in train    
    ...             % 
    Gtrain_test,... % Matrix of user relations between train and test set
    ...             %
    Gtest,...       % Matrix of user relations in test set
    ...             %
    ...             %
    varargin...     % lambda , beta, teta, blah,
)
    % First make model using Gtrain, Ytrain
    % TrainPredicted would just be fitted values
    % Finally, compute TestPredicted using model

end

