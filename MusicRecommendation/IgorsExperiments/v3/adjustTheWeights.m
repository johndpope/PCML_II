function Yout = adjustTheWeights(Y, I, added)

    [pi, pj, pr] = find(I);
    T = length(pi);
    for k=1:T
        i = pi(k);
        j = pj(k);
        pr(k) = exp(added + log(Y(i, j)));
    end
    
    Yout = sparse(pi, pj, pr, size(Y, 1), size(Y, 2));
    
end

