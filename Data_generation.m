%%  Data Generation

twoway = 0; % 1 for two way sparse estimation; 0 for row sparse only

p = 100; n = 50; 
k = 50;
r_true = 8; s1_true = 10; s2_true = 15;

X = randn(n,p); E = randn(n,k);

Utrue = zeros(p,r_true); Utrue(randsample(p,s1_true), :) = randn(s1_true,r_true);
if twoway == 0
    Vtrue = randn(k,r_true); 
else
    Vtrue = zeros(k,r_true); Vtrue(randsample(k,s2_true), :) = randn(s2_true,r_true);
end

Theta0 = Utrue * Vtrue'; % true Theta

Y = X * Theta0 + E;
f_obj = @(U,V) sum(sum((Y - X*U*V').^2))/2/n;

% rank estimation
Xpseudo = pinv(X'*X);
PY = X*Xpseudo*X'*Y;
sin_PY = svd(PY);
sin_thre = ( sqrt(2*k) + sqrt(2*min(n,p)) ) * median(svd(Y)) / sqrt(max(n,k)); 
r_est = sum(sin_PY > sin_thre/1.5);
r = r_est;
% r = r_true; % just use the true rank
