% CSC 578 Project 1 Vicky Lee
% Sample Try
CSC578_Project1([1 2 3; 2 4 6],[0 1 1], [2 2 1], 3, 1, 0.01)

% iris data
iris_trn = iris(:,1:4);
iris_trn = transpose(iris_trn);

iris_trnAns = iris(:,5:7);
iris_trnAns = transpose(iris_trnAns);

% SGD(iris_trn, iris_trnAns, nodeLayers, numEpochs, batchSize, eta)
CSC578_Project_1(iris_trn, iris_trnAns, [4 20 3], 100, 10, 0.1)

% MNIST data
load('mnistTrn.mat')
CSC578_Project_1(trn, trnAns, [784 30 10], 30, 10, 3.0)

% xor data
xor_trn= xor(:,1:2);
xor_trn = transpose(xor_trn);

xor_trnAns = xor(:,3);
xor_trnAns = transpose(xor_trnAns);

CSC578_Project_1(xor_trn, xor_trnAns, [2 2 1], 10, 4, 0.1)
CSC578_Project_1(xor_trn, xor_trnAns, [2 2 1], 10, 1, 0.1)
CSC578_Project_1(xor_trn, xor_trnAns, [2 3 2 1], 20, 1, 0.1)

