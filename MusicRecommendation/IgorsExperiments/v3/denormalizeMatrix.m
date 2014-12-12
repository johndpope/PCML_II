function [Ytr0, Yte0] = denormalizeMatrix(Ytr1, Yte1, Itr, Ite, meanY, stdY)
    
    [pi, pj, pr] = find(Itr);
    T = length(pi);
    for k=1:T
        pr(k) = stdY * (Ytr1(pi(k), pj(k)) + meanY);
    end
    Ytr0 = sparse(pi, pj, pr, size(Itr, 1), size(Itr, 2));
    
    [pi, pj, pr] = find(Ite);
    T = length(pi);
    for k=1:T
        pr(k) = stdY * (Yte1(pi(k), pj(k)) + meanY);
    end
    Yte0 = sparse(pi, pj, pr, size(Ite, 1), size(Ite, 2)); 
    
end
