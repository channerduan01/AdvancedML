%% Digit recognition


%% Init
clear;
close all;
clc;

%% Load Train
TrainData = csvread('train.csv', 1, 0);

%% Load Test
% TestData = csvread('train.csv', 1, 0);

%% Preparation
image_size = 28;
Basey = TrainData(:, 1);
y = 
BaseX = TrainData(:, 2:end);
X = [ones(size(BaseX,1), 1) BaseX];

%% Randomly select to display
sel = randperm(size(X, 1));
sel = sel(1:100);
sample = BaseX(sel(1), :);
fprintf('value: %d\n', y(sel(1)));
colormap(gray);
imagesc(reshape(sample, image_size, image_size)', [-1 1]);

%% Linear Regression
p = image_size*image_size;
cvx_begin quiet
variable w( p+1 );
minimize norm( X*w - Basey )
cvx_end

%% Neural network Regression
[net] = feedforwardnet(5);
[net] = train(net, X', Basey')

%% Compare Regression
fprintf('Linear regression error-train: %d\n', sum((Basey-X*w)).^2);
fprintf('Neural regression error-train: %d\n', sum((Basey-net(X')')).^2);











