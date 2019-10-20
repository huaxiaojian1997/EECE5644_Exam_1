se = zeros(1,100); %  squared-error values
% gamma = zeros(1,N); % w ? N (0, ¦Ã^2*I)
n = 40;
account = -n:1:n;
N = 2*n+1; % the number of gamma
for k=1:N % N gamma
    % B = randi([-14,14],1,28);
    gamma = 10.^account;
    % gamma = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000,10000000,100000];
    % gamma(1,k) = 10^B(k);
    % se = zeros(1,100);
    for i = 1:100 % 100 experiments
    w = [1.5,0.3,-0.5,0.1]'; % w value
    x = zeros(1,10);
    b = zeros(4,10);
    for j = 1:10 % Randomly generate 10 X value
        x(j) = -1+2*rand(1,1); % x ? Uniform[-1,1]
        b(:,j) = [x(j).^3, x(j).^2, x(j).^1, x(j).^0]'; % the matrix with X value
    end
    derta = 1; %  v ? N (0,¦Ò^2)
    v = normrnd(0,derta,10,1); % additive noise
    % v = mvnrnd(0,derta);
    y = (w'*b)'+v; % the matrix with Y value
    % y = w'*b;
    
    % the MAP estimate for the parameter vector w
    s1 = zeros(4,4);
    s2 = zeros(4,1);
    for j = 1:10
        s1 = b(:,j)*(b(:,j))'+s1;
        s2 = b(:,j)*y(j,:)+s2;
    end
    s3 = gamma(1,k)^(-2)*eye(4);
    a = (s1/(derta)^2+s3)^(-1);
    b = s2/(derta)^2;
    wMAP = a*b;
    
    % the squared L2 distance between the true parameter vector and this estimate
    se(k,i) = (w'*w-wMAP'*wMAP)^2;
    % for j = 1:4
    %      c(1,i) = (w(j,1)-wMAP(j,1))^2+c(1,i);
    %     se(k,i) = ((w(j,1))'*w(j,1)-(wMAP(j,1))'*wMAP(j,1))^2;
    % end
    end
end

% the minimum, 25%, median, 75%, and maximum values of these squared-error values
d = sort(se,2);
minimum = d(:,1);
quarter = d(:,25);
median = d(:,50);
threequarter = d(:,75);
maximum = d(:,100);

% show the minimum, 25%, median, 75%, and maximum values of these squared-error values
disp('The minimum values of these squared-error values are:');
disp(minimum);
disp('The 25% values of these squared-error values are:');
disp(quarter);
disp('The median values of these squared-error values are:');
disp(median);
disp('The 75% values of these squared-error values are:');
disp(threequarter);
disp('The maximum values of these squared-error values are:');
disp(maximum);

% plot the minimum, 25%, median, 75%, and maximum values of these squared-error values
plot(gamma,minimum,'+r');  hold on, set(gca,'xscale','log'),set(gca,'yscale','log')
plot(gamma,quarter,'+b');  hold on, set(gca,'xscale','log'),set(gca,'yscale','log')
plot(gamma,median,'or');  hold on, set(gca,'xscale','log'),set(gca,'yscale','log')
plot(gamma,threequarter,'ob');  hold on, set(gca,'xscale','log'),set(gca,'yscale','log')
plot(gamma,maximum,'xr');  hold on, set(gca,'xscale','log'),set(gca,'yscale','log')
legend('minimum values','25% values','median values','75% values','maximum values'), 
title(' squared-error valuesfor the MAP estimator for 30 values of ¦Ã'),
xlabel('gamma'), ylabel(' squared-error values'),