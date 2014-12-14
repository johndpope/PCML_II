function [Ytrain, Ytest] = getSplitStrong(Y, seed, trainRatio)

    setSeed(seed);

    numOfUsers = size(Y, 1);
    rp = randperm(numOfUsers);
    
    Ytrain = Y;
    Ytest = Y;
    for i=1:numOfUsers
        u = rp(i);
        if (i < numOfUsers * trainRatio)
            Ytest(u,:) = 0;
        else
            Ytrain(u,:) = 0;
        end
    end
end
