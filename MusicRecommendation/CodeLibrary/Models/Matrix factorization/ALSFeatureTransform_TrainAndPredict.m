function [ TrainPredicted, TestPredicted ] = ...
    ALSFeatureTransform_TrainAndPredict(...
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

    Itr = (abs(Ytrain) > 1e-9);
    Ite = abs(Ytest) > 1e-9;
    
    YtrainCopy = Ytrain;
    for i=1:size(Ytrain, 1)
        ftr = find(Itr(i,:) > 0);
        if (~isempty(ftr))
            fte = find(Ite(i,:) > 0);
            Ytrain(i,ftr) = ratingTransform(Ytrain(i,ftr), YtrainCopy(i,ftr));
            Ytest(i,fte) = ratingTransform(Ytest(i,fte), YtrainCopy(i,ftr));
        end
    end

%     avg = full(mean(mean(Ytrain(Itr))));
%     Ytrain(Itr) = Ytrain(Itr) - avg;
%     Ytest(Itr)  = Ytest(Itr) - avg;
    
    % See parameters to JavaMatlabLink function for clarification
    [numOfIterations, varargin] = varargGet('Alg_numOfIterations', varargin);
    [numOfFeatures, varargin] = varargGet('Alg_numOfFeatures', varargin);
    [lambda_U, varargin] = varargGet('Alg_lambda_U', varargin);
    [lambda_I, varargin] = varargGet('Alg_lambda_I', varargin);
    
    R = Ytrain;
    N = size(Ytrain, 1);
    D = size(Ytrain, 2);
    K = numOfFeatures;
    
    maxIter = numOfIterations;
    
    aveRating = zeros(1, D);

    for i=1:D
        aveRating(i) = sum(R(:,i)) / sum(Itr(:,i) > 0);
    end
    
    aveRating(isnan(aveRating)) = 0.0;

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
                Ai = M(:,I) * M(:,I)' + lambda_U * length(I) * eye(K);
                Vi = M(:,I)  * valRows{i}';
                U(:,i) = Ai \ Vi;
            end
        end

        for j=1:D
            J = goodCols{j};
            if (~isempty(J))
                Aj = U(:,J) * U(:,J)' + lambda_I * length(J) * eye(K);
                Vj = U(:,J) * valCols{j};
                M(:,j) = Aj \ Vj;
            end
        end
    %    TrainPredicted = Ytrain;
    %    TrainPredicted(find(Itr > 0)) = sum(U(:, IR) .* M(:, JR));

     %   disp(norm(TrainPredicted(find(Itr > 0)) - Ytrain(find(Itr > 0))));
    end
    
    TrainPredicted = Ytrain;
    TrainPredicted(find(Itr > 0)) = sum(U(:, IR) .* M(:, JR));

    
    TestPredicted = Ytest;
    TestPredicted(find(Ite > 0)) = sum(U(:, IRt) .* M(:, JRt));
    
    for i=1:size(Ytrain, 1)
        ftr = find(Itr(i,:) > 0);
        if (~isempty(ftr))
            fte = find(Ite(i,:) > 0);
            TrainPredicted(i,ftr) = DeratingTransform(TrainPredicted(i,ftr), YtrainCopy(i,ftr));
            TestPredicted(i,fte) = DeratingTransform(TestPredicted(i,fte), YtrainCopy(i,ftr));
        end
    end
    
end
