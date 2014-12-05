function [ Y_te ] = DEtransformData( muU, stdU, Y_te, I_te, Y_tr, I_tr)
    for i=1:size(Y_te,1)
        teidx = find(I_te(i,:) > 0);
        Y_te(i,teidx) = Y_te(i,teidx) * stdU(i) + muU(i);
    end
    
    for i=1:size(Y_tr,1)
        tridx = find(I_tr(i,:) > 0);
        teidx = find(I_te(i,:) > 0);
        
        vals = Y_tr(i,tridx);
        Y_te(i,teidx) = DeratingTransform(Y_te(i,teidx), vals);
        
    end
end

