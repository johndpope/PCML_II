function [ meanAll, meanU, meanI, Y_te] = subtractBaseline( Y_te, I_te, Y_tr, I_tr) 
    meanAll = full(mean(mean(Y_tr(I_tr))));
    Y_te(I_te) = Y_te(I_te) - meanAll;
    
    meanU = zeros(size(Y_tr, 1), 1);
    meanI = zeros(size(Y_tr, 2), 1);
        
    for i=1:size(Y_tr, 1)
        idx = find(I_tr(i,:) > 0);
        meanU(i) = full(mean(Y_tr(i, idx)));
        Y_tr(i,idx) = Y_tr(i,idx) - meanU(i);
    end
   

    for i=1:size(Y_tr, 2)
        idx = find(I_tr(:,i) > 0);
        meanI(i) = full(mean(Y_tr(idx, i)));
        Y_tr(idx,i) = Y_tr(idx,i) - meanI(i);
    end
    
    meanU(isnan(meanU)) = 0;
    meanI(isnan(meanI)) = 0;
    
    for i=1:size(Y_te, 1)
        idx = find(I_te(i,:) > 0);
        Y_te(i,idx) = Y_te(i,idx) - meanU(i);
    end
    
    for i=1:size(Y_te, 2)
        idx = find(I_te(:,i) > 0);
        Y_te(idx,i) = Y_te(idx,i) - meanI(i);
    end

end
