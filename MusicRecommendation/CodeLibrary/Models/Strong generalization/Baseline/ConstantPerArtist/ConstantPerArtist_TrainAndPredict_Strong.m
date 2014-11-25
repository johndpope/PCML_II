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
    varargin...     %
)
       
    meanY = full(mean(Ytrain));
    sumB = full(sum(boolean(Ytrain)));
    
    meanY(sumB > 0) = meanY(sumB > 0) * size(Gtrain, 1) ./ sumB(sumB > 0);
    
    TrainPredicted = repmat(meanY, size(Gtrain, 1), 1);
    TestPredicted = repmat(meanY, size(Gtest, 1), 1);
end

