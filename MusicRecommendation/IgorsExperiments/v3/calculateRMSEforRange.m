function [rmse, total] = calculateRMSEforRange(pYte, Yte, I, rangeFr, rangeTo)
    [pi, pj, pr] = find(I);
    T = length(pi);
    total = 0;
    rmse = 0;
    for k=1:T
        i = pi(k);
        j = pj(k);
        if ((rangeFr < pYte(i, j)&&(pYte(i, j) <= rangeTo)))
            total = total + 1;
            rmse = rmse + (pYte(i, j) - Yte(i, j)) * (pYte(i, j) - Yte(i, j));
        end
    end
    if (total > 0)
        rmse = sqrt(rmse / total);
    end
end

