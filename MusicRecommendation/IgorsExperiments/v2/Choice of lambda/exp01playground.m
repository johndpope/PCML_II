load('songTrain.mat');

% we copy the data
Y = Ytrain;
G = Gtrain;

% we remove the empty rows and columns
[Y, G] = removeEmptyRowsAndColumns(Y, getI(Y), G);

% we define the seed
seed = 42;

% parameter of our experiment
numOfIterations = 50;
lambdas = [0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1];
NL = length(lambdas);
seeds = [11, 13, 16, 121, 142, 156, 189, 1122, 18392, 192721];
% number of trials
numOfTrials = length(seeds);

rmseTrVector = zeros(NL, numOfTrials, numOfIterations);
rmseTeVector = zeros(NL, numOfTrials, numOfIterations);
rmseTrVector2 = zeros(NL, numOfTrials, numOfIterations);
rmseTeVector2 = zeros(NL, numOfTrials, numOfIterations);

rmseTrBaselineVector = zeros(NL, numOfTrials);
rmseTeBaselineVector = zeros(NL, numOfTrials);

for k=1:NL
    
    rmseTrainBaseline = 0;
    rmseTestBaseline = 0;
    rmseTrainMyModel = 0;
    rmseTestMyModel = 0;
    
    fprintf('\n==============================\n');
    fprintf('Testing lambda %f.\n', lambdas(k));
    fprintf('==============================\n');
    
    for trial=1:numOfTrials
        seed = seeds(trial)+k;

        fprintf('\nTrial %d\n', trial);

        % we split the data into two sets
        % [Ytr, Yte] = getSplitDataWeak(Y, G, seed, 10);
        [Ytr, Yte] = getSplitDataWeak2(Y, seed, 0.9);

        checkValidity(Ytr);

        % we get the indicator matrices for Ytr and Yte
        Itr = getI(Ytr);
        Ite = getI(Yte);

        fprintf('==============================\n');
        fprintf('Number of ratings in the train dataset: %d.\n', length(find(Itr > 0)));
        fprintf('Number of ratings in the test dataset: %d.\n', length(find(Ite > 0)));
        fprintf('==============================\n');

        % we obtain the baseline predictions
        [pYtr0, pYte0, rmse1, rmse2, rmse3] = getBaseLinePredictions(Ytr, Yte, 0.5);

        fprintf('\n==============================\n');
        fprintf('Baseline prediction - user average: %f.\n', rmse1);
        fprintf('Baseline prediction - artist average: %f.\n', rmse2);
        fprintf('Baseline prediction - weighted user and artist average: %f.\n', rmse3);
        fprintf('==============================\n');
        
        rmseTrBaselineVector(k, trial) = calculateRMSE(pYtr0, Ytr, Itr);
        rmseTeBaselineVector(k, trial) = calculateRMSE(pYte0, Yte, Ite);
        
        rmseTrainBaseline = rmseTrainBaseline + rmseTrBaselineVector(k, trial);
        rmseTestBaseline = rmseTestBaseline + rmseTeBaselineVector(k, trial);

        fprintf('\n==============================\n');
        fprintf('Baseline prediction - weighted user and artist average\n');
        fprintf('Train: %f, test: %f.\n', rmseTrBaselineVector(k, trial), rmseTeBaselineVector(k, trial));
        fprintf('==============================\n');
        
        % we transform the ratings (the ratings should not be changed)

        % option 1
        % [YtrTransformed, lambdaCoef] = transformWithBoxCox(Ytr);
        % YteTransformed = transformWithBoxCoxLambda(lambdaCoef, Yte);

        % option 2
        % [YtrTransformed, lambda] = transformWithBoxCoxRows(Ytr, getI(Ytr));
        % YteTransformed = transformWithBoxCoxLambdaRows(Yte, getI(Yte), lambda);

        % option 3
        YtrTransformed = logaritmizeSparseMatrix(Ytr);
        YteTransformed = logaritmizeSparseMatrix(Yte);

        % we normalize the ratings

        % option 1
        % [YtrTransformed, YteTransformed, meanY, stdY] = normalizeMatrix(YtrTransformed, YteTransformed, Itr, Ite);

        % option 2
        % [YtrTransformed, YteTransformed, meanY] = normalizeMatrixMean(YtrTransformed, YteTransformed, Itr, Ite);

        % option 3
        % we do nothing

        % =============== OUR MODEL HERE ===============
        % Input: YtrTransformed, YteTransformed, Itr, Ite, G
        % Output: Ptr0, Pte0
        fprintf('\n==============================\n');
        fprintf('Output from our model\n');

        % parameters for our model
        numOfLatentFactors = 10;
        lambdaU = lambdas(k);
        lambdaM = lambdas(k);
        lambdaT = 0.01;
        friendshipIncluded = 0;

        fprintf('Number of latent factors: %d.\n', numOfLatentFactors);
        fprintf('Number of iterations: %d.\n', numOfIterations);
        if (friendshipIncluded == 1)
            fprintf('LambdaU: %f, lambdaM: %f, lambdaT: %f.\n', lambdaU, lambdaM, lambdaT);
        else
            fprintf('LambdaU: %f, lambdaM: %f.\n', lambdaU, lambdaM);
        end

        % we run our model

        % method 1
        [Ptr0, Pte0, rmseStart, rmseEnd, rmseBest, U, M, rmseTrVector(k, trial, :), rmseTeVector(k, trial, :), rmseTrVector2(k, trial, :), rmseTeVector2(k, trial, :)] = exp01modelLambdaLS(YtrTransformed, YteTransformed, Itr, Ite, G, seed, numOfLatentFactors, lambdaU, lambdaM, lambdaT, friendshipIncluded, numOfIterations);

        % method 2
        % alg = {'mm', 'cjlin', 'als', 'alsobs', 'prob'};
        % YtrTransformed2 = setNegativeToZero(YtrTransformed);
        % [U, M] = nmf(YtrTransformed2, numOfLatentFactors, alg{3}, numOfIterations, 0);
        % Ptr0 = getPredictions(YtrTransformed, Itr, U, M);
        % Pte0 = getPredictions(YteTransformed, Ite, U, M);

        fprintf('==============================\n');
        % ==============================================
        
        % we denormalize the ratings

        % option 1
        %[Ptr1, Pte1] = denormalizeMatrix(Ptr0, Pte0, Itr, Ite, meanY, stdY);

        % option 2
        % [Ptr1, Pte1] = denormalizeMatrixMean(Ptr0, Pte0, Itr, Ite, meanY);

        % option 3
        Ptr1 = Ptr0;
        Pte1 = Pte0;

        % we detransform the ratings

        % option 1
        % Ptr2 = detransformWithBoxCox(Ptr1, Itr, lambdaCoef);
        % Pte2 = detransformWithBoxCox(Pte1, Ite, lambdaCoef);

        % option 2
        % Ptr2 = detransformWithBoxCoxRows(Ptr1, Itr, lambda);
        % Pte2 = detransformWithBoxCoxRows(Pte1, Ite, lambda);

        % option 3
        Ptr2 = expSparseMatrix(Ptr1, Itr);
        Pte2 = expSparseMatrix(Pte1, Ite);

        fprintf('\n==============================\n');
        fprintf('Prediction with our model\n');
        fprintf('Train: %f, test: %f.\n', calculateRMSE(Ptr2, Ytr, Itr), calculateRMSE(Pte2, Yte, Ite));
        fprintf('==============================\n');

        rmseTrainMyModel = rmseTrainMyModel + calculateRMSE(Ptr2, Ytr, Itr);
        rmseTestMyModel = rmseTestMyModel + calculateRMSE(Pte2, Yte, Ite);
    end

    rmseTrainBaseline = rmseTrainBaseline / numOfTrials;
    rmseTestBaseline = rmseTestBaseline / numOfTrials;

    rmseTrainMyModel = rmseTrainMyModel / numOfTrials;
    rmseTestMyModel = rmseTestMyModel / numOfTrials;

    fprintf('\n==============================\n');
    fprintf('Average prediction with baseline\n');
    fprintf('Train: %f, test: %f.\n', rmseTrainBaseline, rmseTestBaseline);
    fprintf('==============================\n');

    fprintf('\n==============================\n');
    fprintf('Average prediction with our model\n');
    fprintf('Train: %f, test: %f.\n', rmseTrainMyModel, rmseTestMyModel);
    fprintf('==============================\n');
end
