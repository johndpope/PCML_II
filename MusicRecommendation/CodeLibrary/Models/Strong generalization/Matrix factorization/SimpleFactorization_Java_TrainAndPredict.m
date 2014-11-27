function [ TrainPredicted, TestPredicted ] = ...
    SimpleFactorization_Java_TrainAndPredict(...
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
    [numOfIterations, varargin] = varargGet('Alg_numOfIterations', varargin);
    [numOfFeatures, varargin] = varargGet('Alg_numOfFeatures', varargin);
    [lambda, varargin] = varargGet('Alg_lambda', varargin);
    
    cmdArgs = strcat({' SimpleFactorization '},...
        {' numOfFeatures '},{num2str(numOfFeatures)},...
        {' numOfIterations '},{num2str(numOfIterations)},...
        {' lambda '},{num2str(lambda)});
    
    TrainPredicted = JavaMatlabLink(Ytrain, Ytrain,...
        'train.dat', 'test.dat', cmdArgs{1});
    TestPredicted = JavaMatlabLink(Ytrain, Ytest,...
        'train.dat', 'test.dat', cmdArgs{1});
end

