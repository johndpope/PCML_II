%%


clear all;
clc;
load 'Data/songTrain.mat'

G = Gtrain;
Y = Ytrain;

%% Weak

s = [1];
K = 5;

TrRMSE = zeros(K, 1);
TeRMSE = zeros(K, 1);

for sidx=1:length(s)

    for iter=1:K
        [Ytr, Yte] = getSplitDataWeak2(Y, iter, 0.8);
        Itr = Ytr > 0;
        Ite = Yte > 0;
        Ytr(find(Ytr)) = log(Ytr(find(Ytr)));
        Yte(find(Yte)) = log(Yte(find(Yte)));

        [TrPredicted, TePredicted] = ...
            ...%stackAlgorithm(Ytr, Yte, Itr, Ite);
            ...%KNN_Friend_TrainAndPredict(Ytr, Yte, G, Itr, Ite);
            ...%KNN_Mix(Ytr, Yte, G, Itr, Ite, s(sidx));
            modelWeightedLambdaLS(G, Ytr, Yte, Itr, Ite);

        TrRMSE(iter, sidx) = RMSE(TrPredicted, Ytr);
        TeRMSE(iter, sidx) = RMSE(TePredicted, Yte);
        fprintf('Iter %d: %0.4f %0.4f\n', iter, TrRMSE(iter,sidx), TeRMSE(iter,sidx));
    end
end
fprintf('Train: %0.4f  Test: %0.4f\n', mean(mean(TrRMSE)), mean(mean(TeRMSE)));

save('Output/Weak/ALS_Friendship.mat', 'TeRMSE', 'TrRMSE');

%figure;
%scatter(TePredicted(Ite), abs(TePredicted(Ite) - Yte(Ite)));

% figure;
% plot(s, mean(TeRMSE, 1));
% hold on;
% line(s, mean(TeRMSE, 1));
% 
% err = zeros();
% meany = zeros();
% for i=1:size(TePredicted, 1)
%     idx = find(Ite(i,:) > 0);
%     err(i) = mean(abs(TePredicted(i,idx) - Yte(i,idx)));;
%     meany(i) = length(idx);
% end
% 
% figure;
% scatter(meany, err);
% 
% 
% err = zeros();
% meany = zeros();
% t = 1;
% for i=1:size(TePredicted, 2)
%     idx = find(Ite(:,i) > 0);
%     tmp = mean(abs(TePredicted(idx,i) - Yte(idx,i)));;
%     if (isempty(tmp) | isnan(tmp))
%     else
%         err(t) = tmp;
%         meany(t) = length(idx);
%         t = t + 1;
%     end
%     
% end
% 
% figure;
% scatter(meany, err);
% % 
% % less than 3 viewers: error
% % 
% %     0.6638
% % 
% % more than 3
% %     0.6128

%% Strong
K = 5;
TrRMSE = zeros(K, 1);
TeRMSE = zeros(K, 1);

for iter=1:K
    [Ytr, Yte] = getSplitDataStrong(Y, iter, 0.95);
    Itr = Ytr > 0;
    Ite = Yte > 0;
    Ytr(find(Ytr)) = log(Ytr(find(Ytr)));
    Yte(find(Yte)) = log(Yte(find(Yte)));
    
    Gc = G;
    TeU = find(sum(Ite, 2) > 0);
    Gc(TeU, TeU) = 0;
    [TrPredicted, TePredicted] = ...
        ...%stackAlgorithm(Ytr, Yte, Itr, Ite);
        KNN_Friend_TrainAndPredict(Ytr, Yte, Gc, Itr, Ite);
        ...%KNN_Mix(Ytr, Yte, G, Itr, Ite, s(sidx));
        ...%modelWeightedLambdaLS(G, Ytr, Yte, Itr, Ite);

    TrRMSE(iter) = RMSE(TrPredicted, Ytr);
    TeRMSE(iter) = RMSE(TePredicted, Yte);
    fprintf('Iter %d: %0.4f %0.4f\n', iter, TrRMSE(iter), TeRMSE(iter));
end
fprintf('Train: %0.4f  Test: %0.4f\n', mean(mean(TrRMSE)), mean(mean(TeRMSE)));


save('Output/Strong/KNN_0_1.mat', 'TeRMSE', 'TrRMSE');
