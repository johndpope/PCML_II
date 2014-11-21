% Function for running cross-validation given a particular
% recommender algorithm for STRONG generalization.
% Here we assume that we try to predict for new users.
% Required vararg - CV_type:
% 'CV': K-fold-cross-validation, 
%  requires 'CV_k' in varargin
% 'Split': random train-test split
%  requires 0<'Split_p'<1, 'Split_k'    
function [ TrainError, TestError ] = crossValidation_Strong( ...
    Gtrain, ...                     % Sparse friendship graph
    ...                             %
    Ytrain, ...                     % Sparse listen count
    ...                             %
    TrainAndPredictFunction, ...    % Function for training model on
    ...                             % the training data and testing
    ...                             % on the test data
    ...                             % Required output is stated in 
    ...                             % Models/ directory
    ...                             %
    varargin...                     % Various additional arguments to 
    ...                             % be passed in either this function or
    ...                             % TrainAndPredict function
    ...                             % in a typical matlab format:
    ...                             % pairs 'Name', value.
    ...                             % Ex. 'CV_k', 3, 'lambda', 5
    )
    
    [seed,varargin] = varargGet('CV_seed', varargin);
    [CV_type,varargin] = varargGet('CV_type', varargin);
    
    if (strcmp(CV_type, 'CV')) 
        [K,varargin] = varargGet('CV_k', varargin);
    elseif (strcmp(CV_type, 'Split'))
        [P,varargin] = varargGet('CV_p', varargin);
        [K,varargin] = varargGet('CV_k', varargin);
    end

    % Extracting parameters
    N = size(Ytrain, 1); % # of users
    M = size(Ytrain, 2); % # of artists
    
    setSeed(seed);
    % Let's get a train-test split
    if (strcmp(CV_type, 'CV')) 
        TrainIdx = false(K, N);
        TestIdx = false(K, N);
        perm = randperm(N);
        idx = zeros(N, 1);
        nk = floor(N / K);
        for i = 1:K
            idx(perm(((i - 1) * nk + 1):(i * nk))) = i;
        end
        for i = 1:K
            TrainIdx(i,:) = (idx ~= i);
            TestIdx(i,:) = (idx == i);
        end
    elseif (strcmp(CV_type, 'Split'))
        trainPart = floor(N * P);
        TrainIdx = zeros(K, trainPart);
        TestIdx = zeros(K, N - trainPart);
        for i = 1:K
            perm = randperm(N);
            TrainIdx(i,:) = perm(1:trainPart);
            TestIdx(i,:) = perm((trainPart+1):end);
        end
    else
        print 'ERROR - wrong CV type';
        return;
    end
    
    TrainError = zeros(K, 1);
    TestError = zeros(K, 1);
    % Main execution loop
    for foldIdx = 1:K
        [TrainPredicted, TestPredicted] = ...
            TrainAndPredictFunction(...
            Gtrain(TrainIdx(foldIdx,:), TrainIdx(foldIdx,:)),...
            Ytrain(TrainIdx(foldIdx,:),:),...
            Gtrain(TrainIdx(foldIdx,:), TestIdx(foldIdx,:)),...
            Gtrain(TestIdx(foldIdx,:), TestIdx(foldIdx,:)),...
            varargin);
         TrainError(foldIdx) = RMSE(TrainPredicted, ...
                                    Ytrain(TrainIdx(foldIdx,:),:));
         TestError(foldIdx) = RMSE(TestPredicted, ...
                                   Ytrain(TestIdx(foldIdx,:),:));
    end
end

