function [Ytr1, Yte1, meanY] = normalizeMatrixMean(Ytr0, Yte0, Itr, Ite)
    
    % we get all ratings from the training set in one array
    ratingsArrTr = allRatingsInArray(Ytr0, Itr);
    
    meanY = mean(ratingsArrTr);
    
    [pi, pj, pr] = find(Itr);
    T = length(pi);
    for k=1:T
        pr(k) = Ytr0(pi(k), pj(k)) - meanY;
    end
    Ytr1 = sparse(pi, pj, pr, size(Itr, 1), size(Itr, 2));
    
    [pi, pj, pr] = find(Ite);
    T = length(pi);
    for k=1:T
        pr(k) = Yte0(pi(k), pj(k)) - meanY;
    end
    Yte1 = sparse(pi, pj, pr, size(Ite, 1), size(Ite, 2)); 
    
end
