function [ TrainPredicted, TestPredicted ] = ...
    ALS_TrainAndPredict(...
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
    
    R = Ytrain;
    N = size(Ytrain, 1);
    D = size(Ytrain, 2);
    K = numOfFeatures;
    Itr = (abs(R) > 1e-9);
    Ite = abs(Ytest) > 1e-9;
    maxIter = numOfIterations;
    
    aveRating = zeros(1, D);

    for i=1:D
        aveRating(i) = sum(R(:,i)) / sum(Itr(:,i) > 0);
    end
    
    aveRating(isnan(aveRating)) = 0.0;
    aveRating(isinf(aveRating)) = 0.0;

    setSeed(1);
    M = [aveRating; randn(K - 1, D)];
    U = zeros(K, N);

    [IR, JR] = ind2sub(size(Ytrain), find(Itr > 0));
    [IRt, JRt] = ind2sub(size(Ytest), find(Ite > 0));
    
%    [U, M] = nnmf(R, K);
%    U = U';
    
    goodRows = cell(N, 1);
    valRows = cell(N, 1);
    for i=1:N
        goodRows{i} = find(Itr(i,:) > 0);
        valRows{i} = R(i, goodRows{i});
    end
    goodCols = cell(D, 1);
    valCols = cell(D, 1);
    for j=1:D
        goodCols{j} = find(Itr(:,j) > 0);
        valCols{j} = R(goodCols{j}, j);
    end
    
    for iter=1:maxIter
        for i=1:N
            I = goodRows{i};
            if (~isempty(I))
                Ai = M(:,I) * M(:,I)' + lambda * length(I) * eye(K);
                Vi = M(:,I)  * valRows{i}';
                U(:,i) = Ai \ Vi;
            end
        end

        for j=1:D
            J = goodCols{j};
            if (~isempty(J))
                Aj = U(:,J) * U(:,J)' + lambda * length(J) * eye(K);
                Vj = U(:,J) * valCols{j};
                M(:,j) = Aj \ Vj;
            end
        end
    end
    
    TrainPredicted = Ytrain;
    TrainPredicted(find(Itr > 0)) = sum(U(:, IR) .* M(:, JR));
    
    TestPredicted = Ytest;
    TestPredicted(find(Ite > 0)) = sum(U(:, IRt) .* M(:, JRt));
    
 %   shift = 5;
    
 %   TrainPredicted(find(abs(TrainPredicted) > 0 & TrainPredicted < shift)) = shift;
 %   TestPredicted(find(abs(TestPredicted) > 0 & TestPredicted < shift)) = shift;
    
end
