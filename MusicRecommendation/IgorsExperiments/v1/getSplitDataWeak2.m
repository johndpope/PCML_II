function [Ytrain, Ytest] = getSplitDataWeak2(Y, seed, trainRatio)

    setSeed(seed);

    numOfUsers = size(Y, 1);
    numOfItems = size(Y, 2);
    
    [pairsI, pairsJ, pairsV] = find(Y);
    
    T = length(pairsI);
    countU = zeros(numOfUsers, 1);
	countM = zeros(numOfItems, 1);

    for i=1:numOfUsers
        countU(i) = length(find(Y(i, :) > 0));
    end

    for j=1:numOfItems
        countM(j) = length(find(Y(:, j) > 0));
    end
    
    countU1 = zeros(numOfUsers, 1);
	countM1 = zeros(numOfItems, 1);    
    countU2 = zeros(numOfUsers, 1);
	countM2 = zeros(numOfItems, 1);    
    
    p = randperm(T);
    trainT = floor(T * trainRatio);
    testT = T - trainT;
    
    pairsI1 = [];
    pairsJ1 = [];
    pairsV1 = [];

    pairsI2 = [];
    pairsJ2 = [];
    pairsV2 = [];
    
    currentTrain = 0;
    currentTest = 0;
    
    taken = zeros(T, 1);
    for k=1:T
        
        i = pairsI(p(k));
        j = pairsJ(p(k));
        r = pairsV(p(k));
        
        if ((countU1(i) == 0)||(countM1(j) == 0))
            taken(k) = 1;
                
            currentTrain = currentTrain + 1;

            pairsI1 = [pairsI1 i];
            pairsJ1 = [pairsJ1 j];
            pairsV1 = [pairsV1 r];

            countU1(i) = countU1(i) + 1;
            countM1(j) = countM1(j) + 1;
        end
        
    end
    
    for k=1:T
        
        if (taken(k) == 0)
            
            i = pairsI(p(k));
            j = pairsJ(p(k));
            r = pairsV(p(k));
            
            % there should be at least one element in the test set for every
            % item and for every user
            if ((currentTrain < trainT) && ((countU(i) == 1 || countM(j) == 1)||((((countU2(i) > 0) || ((countU(i) - countU1(i)) > 1)) && ((countM2(j) > 0) || ((countM(j) - countM1(j)) > 1))))))
                currentTrain = currentTrain + 1;

                pairsI1 = [pairsI1 i];
                pairsJ1 = [pairsJ1 j];
                pairsV1 = [pairsV1 r];

                countU1(i) = countU1(i) + 1;
                countM1(j) = countM1(j) + 1;

            else

                currentTest = currentTest + 1;

                pairsI2 = [pairsI2 i];
                pairsJ2 = [pairsJ2 j];
                pairsV2 = [pairsV2 r];

                countU2(i) = countU2(i) + 1;
                countM2(j) = countM2(j) + 1;

            end
        end
    end
    
    Ytrain = sparse(pairsI1, pairsJ1, pairsV1, numOfUsers, numOfItems);
    Ytest = sparse(pairsI2, pairsJ2, pairsV2, numOfUsers, numOfItems);
    
end
