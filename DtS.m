function [dist,B] = DtS(A, r, options)
%DtS finds nearest skew-symmetric matrix polynomial B of degree d and rank
% <= r to the matrix polynomial A of degree d. A must have square
% coefficients. Given that A is n by n, r must be in [2,n).
% 
%   Input:
%       A                   :   Matrix polynomial coefficients as an nxnxd
%                               matrix.
%       r                   :   Rank of matrix polynomial B to search for.
%       init_guess (opt.)   :   Initial guess (for testing purposes).
%       epsilon (optional)  :   Sensitivity of termination criteria.
%       max_iter (optional) :   Max nr of iterations.
%
%   Output:
%       dist    :   Distance between A and B, in the Frobenius norm.
%       B       :   Coefficients of nearest matrix polynomial of rank <= r.

    arguments
        A double {mustBeSquare(A)};
        r double {mustBeWithin(r,A)};
        options.init_guess double = NaN;
        options.epsilon double {mustBePositive} = 10^-7;
        options.max_iter double {mustBePositive} = 3000;
    end

    % Termination criteria
    epsilon = options.epsilon;
    term_crit = epsilon*norm(A,"fro");
    N = options.max_iter;

    % Matrix sizes
    s = floor(r/2);
    [n,~,d] = size(A);
    d = d-1;    % d = deg(A)
    dU = floor(d/2);
    dV = ceil(d/2);
    vecA = reshape(A,[],1);

    % Permutation matrices (P vec(X) = vec(X^T))
    P = zeros(n*s,n*s);
    for i=1:n
        for j=1:s
            P((i-1)*s+j, (j-1)*n+i) = 1;
        end
    end

    % Initial guess
    U = options.init_guess;
    if isnan(U)
        U = rand(n,s,dU+1,like=A);
    else
        warning("Initial guess for U not checked for size.")
    end
    
    % dist = zeros(N,1);
    dist = NaN;
    for i = 1:N
        TKU = gen_toeplitz(gen_kronecker(U,P),dV);
        vecV = TKU\vecA;
        V = reshape(vecV,n,s,dV+1);

        TKV = gen_toeplitz(-gen_kronecker(V,P),dU);
        vecU = TKV\vecA;
        U = reshape(vecU,n,s,dU+1);

        B = zeros(size(A));
        for k=1:dU+1
            for l=1:dV+1
                B(:,:,k+l-1) = B(:,:,k+l-1) + ...
                    U(:,:,k)*V(:,:,l).' - V(:,:,l)*U(:,:,k).';
            end
        end

        % dist(i) = norm(A-B,"fro");
        prevdist = dist;
        dist = norm(A-B, "fro");

        % if i>1 && abs(dist(i)-dist(i-1)) < term_crit
        if abs(prevdist-dist) < term_crit
            break;
        end
    end

    % dist = dist(1:i);
end


% NOTE: Theoretically, the main function should run faster with the
% following two functions inlined. Testing shows the opposite, however.

function T = gen_toeplitz(X,d_var)
    [j,k,dX] = size(X);
    dX = dX-1;    % deg(X)

    % number of rows and columns of j by k blocks
    nrows = d_var+dX+1; % deg(A)+1
    ncols = d_var+1;

    T = zeros(j*nrows, k*ncols);
    for c = 1:ncols
        for r = c:c+dX
            T(1+(r-1)*j:r*j,1+(c-1)*k:c*k) = X(:,:,r-c+1);
        end
    end
end


function K = gen_kronecker(U,P)
% When used with V, change sign.
    [n,s,m] = size(U);

    K = zeros(n*n,n*s,m);
    I = eye(n);
    for i = 1:m
        K(:,:,i) = kron(I,U(:,:,i))*P - kron(U(:,:,i),I);
    end
end


function mustBeSquare(M)
    [m,n,~] = size(M);
    if m ~= n
        error("Coefficients must be square.")
    end
end


function mustBeWithin(r,M)
    if 2 > r || r >= length(M)
        error("Rank r must be in [2,n).")
    end
end