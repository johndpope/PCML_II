function [ ret ] = readMatrix( filename, N, M )
    cols = dlmread(filename, ',');
    I = round(cols(:,1) + 1e-9);
    J = round(cols(:,2) + 1e-9);
    V = cols(:,3);
    ret = sparse(I, J, V, N, M);
end

