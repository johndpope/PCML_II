function [ TrainPredicted, TestPredicted ] = ...
    Constant_Java_TrainAndPredict_Strong(...
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
    varargin...     %
)
    % See parameters to JavaMatlabLink function for clarification
    
    TrainPredicted = JavaMatlabLink(Ytrain, Ytrain,...
        'train.dat', 'test.dat', 'GlobalAverage');
    TestPredicted = JavaMatlabLink(Ytrain, Ytest,...
        'train.dat', 'test.dat', 'GlobalAverage');
end

