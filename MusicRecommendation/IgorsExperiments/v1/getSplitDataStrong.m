function [Ytr, Yte, Gtr, Gte] = getSplitDataStrong(Y, G, seed, trainRatio)
    % Test data for Strong generalization
    % keep 10% of users for testing as 'new users'
    % You should decide on your own how many new users you want to test on
    setSeed(seed);
    numOfUsers = size(Y, 1);
    idx = randperm(numOfUsers);
    nTe = floor(numOfUsers * (1 - trainRatio)); 
    idxTe = idx(1:nTe);
    idxTr = idx(nTe+1:end);
    Ytr = Y(idxTr, :);
    Yte = Y(idxTe, :);
    Gtr = G(idxTr, idxTr);
    Gte = G(idxTe, [idxTr idxTe]);
end
