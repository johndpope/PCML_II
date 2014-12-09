function Mout = detransformWithBoxCoxRows(Min, I, lambda)
    [pairsI, pairsJ, pairsV] = find(I);
    L = length(pairsI);
    
    transdat = zeros(L, 1);
    for i=1:L
        transdat(i) = Min(pairsI(i), pairsJ(i));
    end
    
    newVals = zeros(L, 1);
    
    for i=1:L
        if (lambda(pairsI(i)) == 0) 
            newVals(i) = exp(transdat(i)); 
        else
            newVals(i) = (lambda(pairsI(i)) * transdat(i) + 1)^(1/lambda(pairsI(i)));
            %fprintf('Old: %f, new: %f.\n', transdat(i), newVals(i));
        end
    end
    
    Mout = sparse(pairsI, pairsJ, newVals, size(Min, 1), size(Min, 2));
end
