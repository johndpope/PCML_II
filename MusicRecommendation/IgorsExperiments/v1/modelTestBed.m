load('songTrain.mat');

seed = 9;
numD = 10;

reducedUsersRatio = 0.5;
reducedItemsRatio = 0.5;

[Y, G] = reduceDataset(Ytrain, Gtrain, seed, reducedUsersRatio, reducedItemsRatio);

% [Ytr, Yte] = getSplitDataWeak(Y, G, seed, 10);
[Ytr, Yte] = getSplitDataWeak2(Y, seed, 0.9);

numOfLatentFactors = 5;
lambda = 1000;
lambdaT = 1000;
lambdaB = 0.1;
alpha = 0.1;
numOfIterations = 100;

% % we transform the dataset
% [YtrTransformed, Utr, Mtr, Itr] = transformDataset(Ytr);
% [YteTransformed, Ite] = transformDataset2(Yte, Utr, Mtr);
% 
% % we transform the dataset back to the original
% YtrOrig = transformDatasetDecode(YtrTransformed, Utr, Mtr, Itr);
% YteOrig = transformDatasetDecode(YteTransformed, Utr, Mtr, Ite);

[rmse1, rmse2, rmse3] = getBaseLinePredictions(Ytr, Yte);
fprintf('RMSE of baseline models: %f, %f, %f.\n', rmse1, rmse2, rmse3);

friendshipIncludedInTheModel = 1;
[rmseStart, rmseEnd, rmseBest] = modelLambdaBiasExp(Ytr, Yte, G, seed, numOfLatentFactors, lambda, lambdaT, lambdaB, friendshipIncludedInTheModel, alpha, numOfIterations);
fprintf('RMSE of model Lambda with bias.\n');
fprintf('rmseStart: %f, rmseEnd: %f, rmseBest %f.\n\n', rmseStart, rmseEnd, rmseBest);

% friendshipIncludedInTheModel = 0;
% [rmseStart, rmseEnd, rmseBest] = modelLambda(Ytr, Yte, G, seed, numOfLatentFactors, lambda, lambdaT, friendshipIncludedInTheModel, alpha, numOfIterations);
% fprintf('RMSE of model Lambda.\n');
% fprintf('rmseStart: %f, rmseEnd: %f, rmseBest %f.\n\n', rmseStart, rmseEnd, rmseBest);
% 
% friendshipIncludedInTheModel = 1;
% [rmseStart, rmseEnd, rmseBest] = modelLambda(Ytr, Yte, G, seed, numOfLatentFactors, lambda, lambdaT, friendshipIncludedInTheModel, alpha, numOfIterations);
% fprintf('RMSE of model Lambda (with friendships included).\n');
% fprintf('rmseStart: %f, rmseEnd: %f, rmseBest %f.\n\n', rmseStart, rmseEnd, rmseBest);

% fprintf('\nTransformation of the ratings\n');
% 
% % we transform the dataset
% [YtrTransformed, Utr, Mtr, Itr] = transformDataset(Ytr);
% [YteTransformed, Ite] = transformDataset2(Yte, Utr, Mtr);
% 
% friendshipIncludedInTheModel = 0;
% [rmseStart, rmseEnd, rmseBest] = modelLambda(YtrTransformed, YteTransformed, G, seed, numOfLatentFactors, lambda, lambdaT, friendshipIncludedInTheModel, alpha, numOfIterations);
% fprintf('RMSE of model Lambda.\n');
% fprintf('rmseStart: %f, rmseEnd: %f, rmseBest %f.\n\n', rmseStart, rmseEnd, rmseBest);
% 
% friendshipIncludedInTheModel = 1;
% [rmseStart, rmseEnd, rmseBest] = modelLambda(YtrTransformed, YteTransformed, G, seed, numOfLatentFactors, lambda, lambdaT, friendshipIncludedInTheModel, alpha, numOfIterations);
% fprintf('RMSE of model Lambda (with friendships included).\n');
% fprintf('rmseStart: %f, rmseEnd: %f, rmseBest %f.\n\n', rmseStart, rmseEnd, rmseBest);
