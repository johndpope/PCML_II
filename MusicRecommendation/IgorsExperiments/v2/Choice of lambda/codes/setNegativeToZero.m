function B = setNegativeToZero(A)
    [pi, pj, pr] = find(A);
    T = length(pi);
    for k=1:T
        if (pr(k) < 0)
            pr(k) = 0;
        end
    end
    B = sparse(pi, pj, pr, size(A, 1), size(A, 2));
end

