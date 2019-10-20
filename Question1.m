mu(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % mean and covariance of data pdf conditioned on label 1
mu(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
mu(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 3
classPriors = [0.15,0.35,0.5]; thr = [0,cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
actualnumber = [0,0,0,0];
subplot(2,1,1),clf, colorList = 'rbg';
i = 1;
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    actualnumber(i+1) = length(indices);
    fprintf('the actual number of samples that were generated from class %d is %d',i,length(indices));
    fprintf('\n');
    i = i+1;
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(mu(:,l),Sigma(:,:,l),length(indices))';
    subplot(2,1,1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
    legend('class 1','class 2','class 3'),
    title('Data and their ture labels'),
    xlabel('x_1'), ylabel('x_2'), 
end

% classifier gi = ln(p(x|L=i)*p(L=i)), i = 1, 2, 3
% g1 = log(evalGaussian(x,mu(:,1),Sigma(:,:,1)))+log(classPriors(1)); 
% g2 = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))+log(classPriors(2)); 
% g3 = log(evalGaussian(x,mu(:,3),Sigma(:,:,3)))+log(classPriors(3)); 

lambda = [0 0 1;0 1 0;1 0 0]; % loss values
gamma12 = classPriors(2)/classPriors(1); 
discriminantScore12 = log(evalGaussian(x,mu(:,1),Sigma(:,:,1)))-log(evalGaussian(x,mu(:,2),Sigma(:,:,2)));
gamma13 = classPriors(3)/classPriors(1); 
discriminantScore13 = log(evalGaussian(x,mu(:,1),Sigma(:,:,1)))-log(evalGaussian(x,mu(:,3),Sigma(:,:,3)));
gamma21 = classPriors(1)/classPriors(2);  
discriminantScore21 = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));
gamma23 = classPriors(3)/classPriors(2);  
discriminantScore23 = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,3),Sigma(:,:,3)));
gamma31 = classPriors(1)/classPriors(3);  
discriminantScore31 = log(evalGaussian(x,mu(:,3),Sigma(:,:,3)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));
gamma32 = classPriors(2)/classPriors(3);  
discriminantScore32 = log(evalGaussian(x,mu(:,3),Sigma(:,:,3)))-log(evalGaussian(x,mu(:,2),Sigma(:,:,2)));

decision = (discriminantScore12 >= log(gamma12) & discriminantScore13 >= log(gamma13));
ind01 = find(decision==0 & L==1); % L=1 D=2 OR 3
ind11 = find(decision==1 & L==1); % L=1 D=1
decision = (discriminantScore12 < log(gamma12) & discriminantScore13 >= log(gamma13)|discriminantScore12 < log(gamma12) & discriminantScore13 < log(gamma13) & discriminantScore23 >= log(gamma23));
ind21 = find(decision==1 & L==1); % L=1 D=2
decision = (discriminantScore12 >= log(gamma12) & discriminantScore13 < log(gamma13)|discriminantScore12 < log(gamma12) & discriminantScore13 < log(gamma13) & discriminantScore32 > log(gamma32));
ind31 = find(decision==1 & L==1); % L=1 D=3

decision = (discriminantScore21 >= log(gamma21) & discriminantScore23 >= log(gamma23));
ind02 = find(decision==0 & L==2); % L=2 D=1 OR 3
ind22 = find(decision==1 & L==2); % L=2 D=2
decision = (discriminantScore21 < log(gamma21) & discriminantScore23 >= log(gamma23)|discriminantScore21 < log(gamma21) & discriminantScore23 < log(gamma23) & discriminantScore13 >= log(gamma13));
ind12 = find(decision==1 & L==2); % L=2 D=1 
decision = (discriminantScore21 >= log(gamma21) & discriminantScore23 < log(gamma23)|discriminantScore21 < log(gamma21) & discriminantScore23 < log(gamma23) & discriminantScore31 > log(gamma31));
ind32 = find(decision==1 & L==2); % L=2 D=3
 
decision = (discriminantScore31 >= log(gamma31) & discriminantScore32 >= log(gamma32));
ind03 = find(decision==0 & L==3); % L=3 D=2 OR 3
ind33 = find(decision==1 & L==3); % L=3 D=3
decision = (discriminantScore31 < log(gamma31) & discriminantScore32 >= log(gamma32)|discriminantScore31 < log(gamma31) & discriminantScore32 < log(gamma32) & discriminantScore12 >= log(gamma12));
ind13 = find(decision==1 & L==3); % L=3 D=1
decision = (discriminantScore31 >= log(gamma31) & discriminantScore32 < log(gamma32)|discriminantScore31 < log(gamma31) & discriminantScore32 < log(gamma32) & discriminantScore21 >= log(gamma21));
ind23 = find(decision==1 & L==3); % L=3 D=2

fprintf('\n');
fprintf('the confusion matrix is');
fprintf('\n');
cr = [length(ind11),length(ind12),length(ind13);length(ind21),length(ind22),length(ind23);length(ind31),length(ind32),length(ind33)];
disp(cr);

fprintf('the total number of samples misclassified by your classifier is %d',length(ind01)+length(ind02)+length(ind03));
fprintf('\n');

fprintf('the probability of error is %.4f',(length(ind01)+length(ind02)+length(ind03))/N);
fprintf('\n');

subplot(2,1,2);
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
plot(x(1,ind02),x(2,ind02),'or'); hold on,
plot(x(1,ind22),x(2,ind22),'og'); hold on,
plot(x(1,ind03),x(2,ind03),'xr'); hold on,
plot(x(1,ind33),x(2,ind33),'xg'); hold on,
axis equal,
legend('Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Wrong decisions for data from Class 2','Correct decisions for data from Class 2','Wrong decisions for data from Class 3','Correct decisions for data from Class 3'), 
title('Data and their decision labels'),
xlabel('x_1'), ylabel('x_2'), 

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
