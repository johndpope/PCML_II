function P = getPredictions(Y, I, U, M)
    [pi, pj, pr] = find(I);
    T = length(pi);
    for k=1:T
        pr(k) = U(pi(k), :) * M(:, pj(k));
    end
    P = sparse(pi, pj, pr, size(Y, 1), size(Y, 2));
end

