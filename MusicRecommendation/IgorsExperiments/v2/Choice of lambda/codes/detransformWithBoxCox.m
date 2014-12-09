function Mout = detransformWithBoxCox(Min, I, lambda)
    [pairsI, pairsJ, pairsV] = find(I);
    L = length(pairsI);
    
    transdat = zeros(L, 1);
    for i=1:L
        transdat(i) = Min(pairsI(i), pairsJ(i));
    end
    
    newVals = zeros(L, 1);
    
    for i=1:L
        if (lambda == 0) 
            newVals(i) = exp(transdat(i)); 
        else
            newVals(i) = (lambda * transdat(i) + 1)^(1/lambda);
        end
    end
    
    Mout = sparse(pairsI, pairsJ, newVals, size(Min, 1), size(Min, 2));
end
