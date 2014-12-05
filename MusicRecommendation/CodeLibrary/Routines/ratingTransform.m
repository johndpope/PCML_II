function [ Ytest, Ytrain ] = ratingTransform(Ytest, Ytrain)
    if (~isempty(Ytest)) 
        if (nargin < 2) 
            Ytrain = Ytest;
        end

        tmp = Ytrain;
        tmp = sort(tmp);
        totSum = sum(tmp);

        [Ytest, I] = sort(Ytest);
        j = 1;
        curSum = 0;
        for i=1:length(Ytest)
            while(j <= length(tmp) && tmp(j) <= Ytest(i)) 
                curSum = curSum + tmp(j);
                j = j + 1;
            end
            Ytest(i) = curSum / totSum;
        end
        Ytest = Ytest(I);
    end
    Ytest = sqrt(Ytest);
end

