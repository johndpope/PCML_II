function [ Y_te ] = DEtransformData( lambda, muU, stdU, Y_te, I_te, Y_tr, I_tr)
    for i=1:size(Y_te,1)
        teidx = find(I_te(i,:) > 0);
        Y_te(i,teidx) = Y_te(i,teidx) * stdU(i) + muU(i);
    end
    
	Y_te = detransformWithBoxCox(Y_te, I_te, lambda);
    
end

