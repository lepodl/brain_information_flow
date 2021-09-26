function [Ux,Uy,R,res] = velocity_field_2d_tensor(BrainImg,para_model,para_alg)
%VELOCITY_FIELD_2D_TENSOR extracts the velocity field from brain images by 
%a tensor model solved by GADIM/MSS
% Input:
%   BrainImg: an (M+1)*(N+1)*(T+1) tensor (BOLD-fMRI signal)
%             y-, x-, t-direction
%   para_model: model parameters including ...
%       rho: the penalty parameter for the smooth regularization
%       tau: the penalty parameter for the time-continuity
%   para_alg: algorithm parameters including ...
%       alpha,beta: stepsizes
%       omega: relaxation
%       tol: tolerance for stopping rule
%       maxit: maximum number of iterations
% Output:
%   Ux,Uy: M*N*T tensors of the x- and y-components of the velocity field
%   res: residuals
%   by Weiyang Ding @Fudan June 23, 2021

% Estimate the partial derivatives
conv_ker = repmat([1,-1]./4,[2,1,2]);
Dx = convn(BrainImg,conv_ker,'valid');% d/dx
Dy = convn(BrainImg,permute(conv_ker,[2,1,3]),'valid');% d/dy
Dt = convn(BrainImg,permute(conv_ker,[1,3,2]),'valid');% d/dt
% Initialization
res_primal = zeros(8,1);
res_dual = zeros(6,1);
res = zeros(2,para_alg.maxit);
[M,N,T] = size(Dx);% M - y-axis, N - x-axis, T - t-axis
Px = zeros([M,N,T]); Py = zeros([M,N,T]);
Qxx = zeros([M,N,T]); Qxy = zeros([M,N,T]);
Qyx = zeros([M,N,T]); Qyy = zeros([M,N,T]);
Rx = zeros([M,N,T]); Ry = zeros([M,N,T]);
Lxx = zeros([M,N,T]); Lxy = zeros([M,N,T]); Lxt = zeros([M,N,T]);
Lyx = zeros([M,N,T]); Lyy = zeros([M,N,T]); Lyt = zeros([M,N,T]);
% Compute the FFT coefficients
cx = conj(fft([1;-1;zeros(N-2,1)])).';
cy = conj(fft([1;-1;zeros(M-2,1)]));
ct = conj(fft(permute([1;-1;zeros(T-2,1)],[3,2,1])));
Cxyt = ones([M,N,T]).*(1+para_alg.beta*para_alg.beta);
Cxyt = bsxfun(@plus,Cxyt,abs(cx).^2.*(para_alg.beta*para_alg.beta));
Cxyt = bsxfun(@plus,Cxyt,abs(cy).^2.*(para_alg.beta*para_alg.beta));
Cxyt = bsxfun(@plus,Cxyt,abs(ct).^2.*(para_alg.beta*para_alg.beta));
for iter = 1:para_alg.maxit
    % U-update
    Temp = (Dx.*Px + Dy.*Py + Dt)./(Dx.*Dx + Dy.*Dy);
    Temp = sign(Temp).*min(abs(Temp),para_alg.alpha);
    Temp(isnan(Temp)) = 0;
    Ux = Px - Dx.*Temp;
    Uy = Py - Dy.*Temp;
    % V-update
    Temp = (para_model.rho*para_alg.alpha)./sqrt(Qxx.*Qxx+Qxy.*Qxy+Qyx.*Qyx+Qyy.*Qyy);
    Temp = 1- min(Temp,1);
    Vxx = Qxx.*Temp;
    Vxy = Qxy.*Temp;
    Vyx = Qyx.*Temp;
    Vyy = Qyy.*Temp;
    Temp = gradx(Ux) - Vxx; res_dual(1) = norm(Temp(:));
    Temp = grady(Ux) - Vxy; res_dual(2) = norm(Temp(:));
    Temp = gradx(Uy) - Vyx; res_dual(3) = norm(Temp(:));
    Temp = grady(Uy) - Vyy; res_dual(4) = norm(Temp(:));
    % W-update
    Wx = Rx./(1+para_model.tau*para_alg.alpha);
    Wy = Ry./(1+para_model.tau*para_alg.alpha);
%     Wx = max(Rx - para_model.tau*para_alg.alpha, 0);
%     Wy = max(Ry - para_model.tau*para_alg.alpha, 0);
    Temp = gradt(Ux) - Wx; res_dual(5) = norm(Temp(:));
    Temp = gradt(Uy) - Wy; res_dual(6) = norm(Temp(:));
    % lambda-update
    Temp = Ux.*2-Px;
    Lxx = Lxx - (gradx(Temp)-Vxx.*2+Qxx).*para_alg.beta;
    Lxy = Lxy - (grady(Temp)-Vxy.*2+Qxy).*para_alg.beta;
    Lxt = Lxt - (gradt(Temp)-Wx.*2+Rx).*para_alg.beta;
    etax = real(fftn((bsxfun(@times,ifftn(Lxx),cx)+bsxfun(@times,ifftn(Lxy),cy)+bsxfun(@times,ifftn(Lxt),ct))./Cxyt));
    Lxx = (Lxx - gradx(etax).*(para_alg.beta*para_alg.beta))./(1+para_alg.beta*para_alg.beta);
    Lxy = (Lxy - grady(etax).*(para_alg.beta*para_alg.beta))./(1+para_alg.beta*para_alg.beta);
    Lxt = (Lxt - gradt(etax).*(para_alg.beta*para_alg.beta))./(1+para_alg.beta*para_alg.beta);
    Temp = Uy.*2-Py;
    Lyx = Lyx - (gradx(Temp)-Vyx.*2+Qyx).*para_alg.beta;
    Lyy = Lyy - (grady(Temp)-Vyy.*2+Qyy).*para_alg.beta;
    Lyt = Lyt - (gradt(Temp)-Wy.*2+Ry).*para_alg.beta;
    etay = real(fftn((bsxfun(@times,ifftn(Lyx),cx)+bsxfun(@times,ifftn(Lyy),cy)+bsxfun(@times,ifftn(Lyt),ct))./Cxyt));
    Lyx = (Lyx - gradx(etay).*(para_alg.beta*para_alg.beta))./(1+para_alg.beta*para_alg.beta);
    Lyy = (Lyy - grady(etay).*(para_alg.beta*para_alg.beta))./(1+para_alg.beta*para_alg.beta);
    Lyt = (Lyt - gradt(etay).*(para_alg.beta*para_alg.beta))./(1+para_alg.beta*para_alg.beta);
    % P-update
    Temp = Ux-Px+etax.*para_alg.beta; res_primal(1) = norm(Temp(:));
    Px = Px + Temp.*para_alg.omega;
    Temp = Uy-Py+etay.*para_alg.beta; res_primal(2) = norm(Temp(:));
    Py = Py + Temp.*para_alg.omega;
    % Q-update
    Temp = Vxx-Qxx-Lxx.*para_alg.beta; res_primal(3) = norm(Temp(:));
    Qxx = Qxx + Temp.*para_alg.omega;
    Temp = Vxy-Qxy-Lxy.*para_alg.beta; res_primal(4) = norm(Temp(:));
    Qxy = Qxy + Temp.*para_alg.omega;
    Temp = Vyx-Qyx-Lyx.*para_alg.beta; res_primal(5) = norm(Temp(:));
    Qyx = Qyx + Temp.*para_alg.omega;
    Temp = Vyy-Qyy-Lyy.*para_alg.beta; res_primal(6) = norm(Temp(:));
    Qyy = Qyy + Temp.*para_alg.omega;
    % R-update
    Temp = Wx-Rx-Lxt.*para_alg.beta; res_primal(7) = norm(Temp(:));
    Rx = Rx + Temp.*para_alg.omega;
    Temp = Wy-Ry-Lyt.*para_alg.beta; res_primal(8) = norm(Temp(:));
    Ry = Ry + Temp.*para_alg.omega;
    % Check convergence
    res(1,iter) = norm(res_primal,'inf');
    res(2,iter) = norm(res_dual,'inf');
    if norm(res(:,iter),'inf') < para_alg.tol
        break
    end
end

R.res = Dx.*Ux + Dy.*Uy + Dt;
R.grad = sqrt(gradx(Ux).^2 + grady(Uy).^2 + gradx(Uy).^2 + grady(Uy).^2);

% Sub-functions
    function g = gradx(u)
        g = u - u(:,[2:N,1],:);
    end

    function g = grady(u)
        g = u - u([2:M,1],:,:);
    end

    function g = gradt(u)
        g = u - u(:,:,[2:T,1]);
    end

end

