 function [ Ytest ] = DeratingTransform(Ytest, Ytrain)
    if (~isempty(Ytest)) 
        Ytest = Ytest .* Ytest;
        [Ytest, I] = sort(Ytest);
        j = 1;

        origVals = ratingTransform(Ytrain, Ytrain);
        Ytrain = sort(Ytrain);

        for i=1:length(Ytest)
            while(j <= length(Ytrain) && Ytest(i) > origVals(j)) 
                j = j + 1;
            end
            if (Ytest(i) < origVals(1))
                Ytest(i) = 0;
            elseif (j > length(Ytrain))
                Ytest(i) = Ytrain(length(Ytrain));
            else
                dleft = Ytest(i) - origVals(j - 1);
                dright = origVals(j) - Ytest(i);
                Ytest(i) = (Ytrain(j - 1) * dright + Ytrain(j) * dleft) / (dright + dleft);
            end
        end
        Ytest = Ytest(I);
    end
end

