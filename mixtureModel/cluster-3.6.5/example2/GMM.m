function x = GMM(N,Pi,mu1,mu2,mu3,R1,R2,R3)

pi1 = Pi(1);
pi2 = Pi(2);
pi3 = Pi(3);

[V,D] = eig(R1);
A1 = V*sqrt(D);

[V,D] = eig(R2);
A2 = V*sqrt(D);

[V,D] = eig(R3);
A3 = V*sqrt(D);

x1 = A1*randn(2,N) + mu1*ones(1,N);

x2 = A2*randn(2,N) + mu2*ones(1,N);

x3 = A3*randn(2,N) + mu3*ones(1,N);

SwitchVar = ones(2,1)*random('Uniform',0,1,1,N);
SwitchVar1 = SwitchVar<pi1;
SwitchVar2 = (SwitchVar>=pi1)&(SwitchVar<(pi1+pi2));
SwitchVar3 = SwitchVar>=(pi1+pi2);

x = SwitchVar1.*x1 + SwitchVar2.*x2 + SwitchVar3.*x3;

