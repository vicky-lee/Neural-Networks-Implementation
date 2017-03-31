% CSC 578 Project 2 Vicky Lee
% iris data
iris_trn = iris(:,1:4);
iris_trn = transpose(iris_trn);

iris_trnAns = iris(:,5:7);
iris_trnAns = transpose(iris_trnAns);

% CSC578_Project_2(inputs, targets, split, nodeLayers, numEpochs, batchSize, eta, TransferFnc, CostFunction, Mom, Reg, EarlyStop, LearnedWeight, LearnedBias)
CSC578_Project_2(iris_trn, iris_trnAns, [80 10 10], [4 20 3], 40, 10, 0.1, 'Sigmoid', 'CrossEntropyCost',0.3, 5, 'False', 'Null', 'Null')
CSC578_Project_2(iris_trn, iris_trnAns, [80 10 10], [4 20 3], 40, 10, 0.1, 'ReLU', 'CrossEntropyCost', 0.3, 5, 'False', 'Null', 'Null')
CSC578_Project_2(iris_trn, iris_trnAns, [80 10 10], [4 20 3], 40, 10, 0.1, 'ReLU', 'CrossEntropyCost', 0, 5, 'False', 'Null', 'Null')

% MNIST data
load('mnistTrn.mat')

CSC578_Project_2(trn, trnAns, [80 10 10], [784 30 10], 30, 10, 3.0, 'Sigmoid', 'QuadraticCost', 0.3, 5, 'False', 'Null', 'Null')
CSC578_Project_2(trn, trnAns, [80 10 10], [784 30 10], 30, 10, 3.0, 'Softmax', 'LogLikelihoodCost', 0, 5, 'False', 'Null', 'Null')
CSC578_Project_2(trn, trnAns, [80 10 10], [784 30 10], 30, 10, 1.0, 'Softmax', 'LogLikelihoodCost', 0.3, 5, 'False', 'Null', 'Null')

% xor data
xor_trn= xor(:,1:2);
xor_trn = transpose(xor_trn);

xor_trnAns = xor(:,3);
xor_trnAns = transpose(xor_trnAns);

CSC578_Project_2(xor_trn, xor_trnAns, [50 25 25], [2 3 2 1], 20, 1, 0.1, 'Sigmoid', 'CrossEntropyCost', 0.3, 5, 'False', 'Null', 'Null')
CSC578_Project_2(xor_trn, xor_trnAns, [50 25 25], [2 3 2 1], 20, 1, 0.1, 'Tanh', 'CrossEntropyCost', 0.3, 5, 'False', 'Null', 'Null')
CSC578_Project_2(xor_trn, xor_trnAns, [50 25 25], [2 3 2 1], 20, 1, 0.1, 'ReLU', 'CrossEntropyCost', 0.3, 5, 'False', 'Null', 'Null')

% Use learned network of weights and biases as the input of the next run
[learned_weight,learned_bias] = CSC578_Project_2(iris_trn, iris_trnAns, [80 10 10], [4 20 3], 40, 10, 0.1, 'Sigmoid', 'CrossEntropyCost',0.3, 5, 'False', 'Null','Null')
CSC578_Project_2(iris_trn, iris_trnAns, [80 10 10], [4 20 3], 40, 10, 0.1, 'Sigmoid', 'CrossEntropyCost',0.3, 5, 'False', learned_weight, learned_bias)

% Test Early stopping
CSC578_Project_2(iris_trn, iris_trnAns, [80 10 10], [4 20 3], 500, 10, 0.1, 'Sigmoid', 'CrossEntropyCost',0.3, 5, 'True', 'Null', 'Null')

