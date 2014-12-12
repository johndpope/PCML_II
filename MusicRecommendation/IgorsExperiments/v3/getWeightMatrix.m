function weight = getWeightMatrix(Ytr, Itr, weightingRate)
    [pi, pj, pr] = find(Itr);
    T = length(pi);
    for k=1:T
        pr(k) = 1 + weightingRate * (Ytr(pi(k), pj(k)) - 1);
    end
    weight = sparse(pi, pj, pr, size(Itr, 1), size(Itr, 2));
end

