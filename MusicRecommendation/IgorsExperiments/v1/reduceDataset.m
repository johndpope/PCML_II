function [Yreduced, Greduced] = reduceDataset(Y, G, seed, reducedUsersRatio, reducedItemsRatio)
    
    setSeed(seed);
    
    numOfUsers = size(Y, 1);
    numOfItems = size(Y, 2);
    
    numOfUsersReduced = floor(numOfUsers * reducedUsersRatio);
    numOfItemsReduced = floor(numOfItems * reducedItemsRatio);
    
    idu = randperm(numOfUsers);
    iduReduced = idu(1:numOfUsersReduced);
    
    idm = randperm(numOfItems);
    idmReduced = idm(1:numOfItemsReduced);
    
    Yreduced = Y(iduReduced, idmReduced);
    Greduced = G(iduReduced, iduReduced);
    
end
