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
       
    meanY = full(mean(Ytrain));
    sumB = full(sum(boolean(Ytrain)));
    % This is now mean for every artist
    meanY(sumB > 0) = meanY(sumB > 0) * size(Gtrain, 1) ./ sumB(sumB > 0);
    
    % A now keeps the index of artist for each non-zero position
    [U, A] = ind2sub(size(Ytrain), find(Ytrain(:) > 0));
    TrainPredicted = sparse(size(Ytrain, 1), size(Ytrain, 2));
    % Now fill each non-zero value with the mean for the appropriate artist
    TrainPredicted(Ytrain(:) > 0) = meanY(A);
    
    % Same for test
    [U, A] = ind2sub(size(Ytest), find(Ytest(:) > 0));
    TestPredicted = sparse(size(Ytest, 1), size(Ytest, 2));
    TestPredicted(Ytest(:) > 0) = meanY(A);
    
end

