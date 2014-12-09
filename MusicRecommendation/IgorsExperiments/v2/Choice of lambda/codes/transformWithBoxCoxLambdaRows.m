function Mout = transformWithBoxCoxLambdaRows(Min, I, lambda)
    
    numOfUsers = size(Min, 1);
    numOfItems = size(Min, 2);
    
    Mout = Min;
    
    for i=1:numOfUsers
        % we take all values from the i-th row
        [pi, pj, pr] = find(I(i, :) > 0);
        T = length(pj);
        transdat = zeros(T, 1);
        for k=1:T
            transdat(k) = Min(i, pj(k));
        end
        nv = boxcox(lambda(i), transdat);
        for k=1:T
            Mout(i, pj(k)) = nv(k);
        end
    end
    
end
