function areSame = sameMatrices(Y0, Y1, I)
    
    areSame = 1;
    [pi, pj, pr] = find(I);
    T = length(pi);
    for k=1:T
        if ((Y0(pi(k), pj(k)) - Y1(pi(k), pj(k))) > 0.000001)
            areSame = 0;
            break;
        end
    end

end

