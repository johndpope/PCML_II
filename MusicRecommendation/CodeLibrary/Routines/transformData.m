function [ muU, stdU, Y_te ] = transformData( Y_te, I_te, Y_tr, I_tr )
    
    for i=1:size(Y_tr,1)
        tridx = find(I_tr(i,:) > 0);
        teidx = find(I_te(i,:) > 0);
        
        vals = Y_tr(i,tridx);
        
        Y_tr(i,tridx) = ratingTransform(Y_tr(i,tridx), vals);
        Y_te(i,teidx) = ratingTransform(Y_te(i,teidx), vals);
        
    end

    muU = zeros(size(Y_tr, 1), 1);
    stdU = zeros(size(Y_tr, 1), 1);
    for i=1:size(Y_tr,1)
        tridx = find(I_tr(i,:) > 0);
        muU(i) = mean(Y_tr(i, tridx));
        stdU(i) = std(Y_tr(i, tridx));
    end
    muU(isnan(muU)) = 0;
    stdU(isnan(stdU)) = 1;
    stdU(abs(stdU) < 1e-9) = 1;
    for i=1:size(Y_te,1)
        teidx = find(I_te(i,:) > 0);
        Y_te(i, teidx) = Y_te(i,teidx) - muU(i);
        Y_te(i, teidx) = Y_te(i,teidx) / stdU(i);
    end
end

