%% GDT algorithm
%  gradient descent with hard thresholding

addpath([pwd,'/glmnet_matlab'])

s1 = 2 * s1_true; s2 = 2 * s2_true; % can also choose using (cross) validation


%  step 1: calculate Separate Lasso estimation, as initialization

Theta_SL = zeros(p,k); lambda = sqrt(log(p)/n)/2;
for j = 1:k

    options = glmnetSet();
    options.standardize = false;
    options.intr = false;
    options.standardize_resp = false;
    options.lambda = lambda;
    options.thre = 1e-3;
    fitt = glmnet(X, Y(:,j),[],options);
    beta = fitt.beta;
    Theta_SL(:,j) = beta;
    
end

err_SL = norm(Theta_SL - Theta0)/norm(Theta0);

% find a factorization UV' ~ Theta_SL

Theta_start = Theta_SL;

[UU,SS,VV] = svd(Theta_start,'econ');
UU = UU(:,1:r); SS = SS(1:r,1:r); VV = VV(:,1:r);
Theta_SVD = UU*SS*VV';
U1 = UU * SS^0.5; V1 = VV * SS^0.5;

U1_h = hard_thre(U1,s1); U1 = U1_h;
if twoway == 1
    V1_h = hard_thre(V1,s2); V1 = V1_h;
end

Theta_1 = U1 * V1';
err_initial = norm(Theta_1 - Theta0)/norm(Theta0);


%% step 2: begin non-convex optimization
%  gradient descent and hard thresholding

U = U1; V = V1;

iter = 0; eta = 0.01; distance = 1;
rho = 1;
OBJ = []; ERR = [];
while distance > 1e-3
    
    iter = iter + 1;
    if iter > 10000
        break
    end
    
    U_old = U; V_old = V;
    
    % GD on U
    grad_U = -X'*(Y-X*U*V')*V / n;
    U = U - eta * grad_U - eta * rho*U*(U'*U-V'*V);
    U = hard_thre(U,s1);
    
    
    % GD on V
    grad_V = -(Y'-V*U'*X')*X*U / n;
    V = V - eta * grad_V - eta * rho*V*(V'*V-U'*U);
    if twoway == 1
        V = hard_thre(V,s2);
    end
    
    distance = norm(U-U_old) + norm(V-V_old);
    
    obj = f_obj(U,V);
    OBJ(iter) = f_obj(U,V);
        
    %if mod(iter,100) == 0
    %    fprintf(' %d ',iter); fprintf(' distance=%6.4f ',distance); fprintf(' obj=%6.4f \n',obj);
    %end
    
end

Theta_nonconvex = U * V';
err_nonconvex = norm(Theta_nonconvex - Theta0)/norm(Theta0);

