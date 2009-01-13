
close all
clear all

N=500;

R1 = [ 1, 0.1; 0.1, 1];
mu1 = [2, 2]';

R2 = [ 1, -0.1; -0.1, 1];
mu2 = [-2, -2]';

R3 = [ 1, 0.2; 0.2, 0.5];
mu3 = [5.5, 2]';

Pi = [0.4, 0.4, 0.2];

% Generate Data from Gaussian mixture model
x1 = GMM(N,Pi,mu1,mu2,mu3,R1,R2,R3); % training
y1 = GMM(N,Pi,mu1,mu2,mu3,R1,R2,R3); % testing

axis([-8,10,-7,7])
plot(x1(1,:),x1(2,:),'bo');
title('Scatter Plot of Multimodal Traning Data for Class 0')
xlabel('first component')
ylabel('second component')
hold on


R1 = [ 1, 0.1; 0.1, 1];
mu1 = [-2, 2]';
                                                                                
R2 = [ 1, -0.1; -0.1, 1];
mu2 = [2, -2]';
                                                                                
R3 = [ 1, 0.2; 0.2, 0.5];
mu3 = [-5.5, 2]';
                                                                                
Pi = [0.4, 0.4, 0.2];
                                                                                
% Generate Data from Gaussian mixture model
x2 = GMM(N,Pi,mu1,mu2,mu3,R1,R2,R3); % training
y2 = GMM(N,Pi,mu1,mu2,mu3,R1,R2,R3); % testing
                                                                                
plot(x2(1,:),x2(2,:),'rx');
title('Scatter Plot of Multimodal Training Data for Class 1')
xlabel('first component')
ylabel('second component')
                                                                                

% Concatenate testing data from two distributions
y = [y1,y2];

figure
plot(y(1,:),y(2,:),'rx');
title('Scatter Plot of Multimodal Testing Data')
xlabel('first component')
ylabel('second component')



% Save data to files
x1 = x1';
save TrainingData1 x1 /ascii

x2 = x2';
save TrainingData2 x2 /ascii

y = y';
save TestingData y /ascii




