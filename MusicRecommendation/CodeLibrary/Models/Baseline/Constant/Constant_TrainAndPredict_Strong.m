function [ TrainPredicted, TestPredicted ] = ...
    Constant_TrainAndPredict_Strong(...
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
       
    meanVal = mean(mean(Ytrain(Ytrain > 0)));
    TrainPredicted = Ytrain;
    TrainPredicted(TrainPredicted > 0) = meanVal;
    TestPredicted  = Ytest;
    TestPredicted(TestPredicted > 0) = meanVal;
end

