function [Mout, lambda] = transformWithBoxCoxRows(Min, I)
    
    numOfUsers = size(Min, 1);
    numOfItems = size(Min, 2);
    
    lambda = zeros(numOfUsers, 1);
    Mout = Min;
    
    for i=1:numOfUsers
        % we take all values from the i-th row
        [pi, pj, pr] = find(I(i, :) > 0);
        T = length(pj);
        transdat = zeros(T, 1);
        for k=1:T
            transdat(k) = Min(i, pj(k));
        end
        [nv, lambda(i)] = boxcox(transdat);
        for k=1:T
            Mout(i, pj(k)) = nv(k);
            %fprintf('Setting %d %d\n', i, pj(k));
        end
    end
    
end
