% CSC 578 Project 2 Vicky Lee
function [learned_weight, learned_bias] = CSC578_Project_2(inputs, targets, split, nodeLayers, numEpochs, batchSize, eta, TransferFnc, CostFunction, Mom, Reg, EarlyStop, LearnedWeight, LearnedBias)

% Turn string input parameter of TransferFuction to function
TransferPrime = strcat(TransferFnc,'Prime');
TransferFnc = str2func(TransferFnc);
TransferPrime = str2func(TransferPrime);

% Shuffle the data
order = randperm(size(inputs,2));
inputs = inputs(:,order);
targets = targets(:,order);

% Partition inputs & targets into training, validation, and test sets
partition1 =  (split(1)/100)*size(inputs,2) ;
partition2 =  ((split(1)+split(2))/100)*size(inputs,2) ;

training_x = inputs(:,1:partition1);
training_y = targets(:,1:partition1);

validation_x = inputs(:, partition1+1:partition2);
validation_y = targets(:,partition1+1:partition2);

test_x = inputs(:,partition2+1:size(inputs,2));
test_y = targets(:,partition2+1:size(inputs,2));

% Initialize weights and biases with random numbers of normal distribution
% with mean 0 and standard deviation 1/sqrt(number of input neurons)
% When parameter, LearnedBias is 'Null', initialize random biases
if strcmp(LearnedBias, 'Null')
    bias = cell(1,length(nodeLayers)-1);
    for i = 1:length(nodeLayers)-1;
        bias{i} = randn(nodeLayers(i+1),1);
    end
% When learned network is passed on as LearnedBias, replace initial biases as
% LernedBias
else
    bias = LearnedBias;
end

% When parameter, LearnedWeight is 'Null', initialize random weights
if strcmp(LearnedWeight, 'Null')
    weight = cell(1,length(nodeLayers)-1);
    for i = 1:length(nodeLayers)-1;
        weight{i} = randn(nodeLayers(i+1),nodeLayers(i))/sqrt(nodeLayers(i)) ;
    end
% When learned network is passed on as LearnedWeight, replace inital weights as
% LernedWeight
else
    weight = LearnedWeight;
end

% Initialize t(Epoch iteration number), list of accuracies & costs for
% training, validation, and test sets, and EarlyStop indicator
% Indicator is 'True' when EarlyStop condition is not reached and turns
% 'False' when EarlyStop condition is satisfied
t = 0;
acc_train = [];
acc_val = [];
acc_test = [];
cost_train = [];
cost_val = [];
cost_test = [];
indicator = 'True';

% Print the header lines for results over epochs
fprintf('     |          TRAIN              ||            VALIDATION       ||           TEST              \n')
fprintf('--------------------------------------------------------------------------------------------------\n')
fprintf('Ep   | Cost  |     Corr     | Acc  || Cost  |     Corr     | Acc  || Cost   |     Corr     | Acc \n')
fprintf('--------------------------------------------------------------------------------------------------\n')

% Repeat epoch while iteration is less than or equal to numEpochs and 
% EarlyStop indicator is 'True'
while t <= numEpochs-1 & strcmp(indicator,'True')
    
    % Create variables t(iteration number for numEpochs), total1,2,3( number 
    % of correctly classified samples), and cost1,2,3 for each training, 
    % validtion and test sets
    t = t+1;
    total1=0;
    total2=0;
    total3=0;
    cost1=0;
    cost2=0;
    cost3=0;
    
    % Shuffle the training data 
    order = randperm(size(training_x,2));
    training_x = training_x(:,order);
    training_y = training_y(:,order);
 
    % Split the training data, inputs and targets into mini batches
    % by batchSize
    mini_x = cell(1,int16(size(training_x,2)/batchSize));
    mini_y = cell(1,int16(size(training_x,2)/batchSize));
    n=0;
    for i=1:batchSize:size(training_x,2)-batchSize+1;
        n = n+ 1;
        mini_x{n} = training_x(:,i:i+batchSize-1);
        mini_y{n} = training_y(:,i:i+batchSize-1);
    end
    

    % Prev_nabla( delta of period n-1 ) is initialized for momentum term
    % in weight update
    for i = 1:length(nodeLayers)-1;
        prev_nabla{i} = zeros(nodeLayers(i+1),nodeLayers(i));
    end
    
    % Mini-Batch SGD
    % Update weights and biases through all mini batches
    
    for k=1:length(mini_x);
        
        nabla_b = cell(1,length(bias));
        nabla_w = cell(1,length(weight));
            
        % Feedforward       
        activation = mini_x{k};
        activations = cell(1,length(nodeLayers));
        activations{1} = mini_x{k};
        zs = cell(1,length(bias));
        
        % When TransferFnc is 'Softmax', Sigmoid activation function is used for 
        % all layers except the output layer where Softmax is used
        if strcmp(func2str(TransferFnc),'Softmax')
            for i=1:length(bias)-1;
                z = weight{i} * activations{i} + repmat(bias{i},1,batchSize);
                zs{i} = z;
                activation = Sigmoid(z);
                activations{i+1} = activation;
            end
            z = weight{length(bias)} * activations{length(bias)} + repmat(bias{length(bias)},1,batchSize);
            zs{length(bias)} = z;
            activation = TransferFnc(z);
            activations{length(bias)+1} = activation;
        % When TransferFnc is other than 'Softmax', given TransferFnc is used    
        else
            for i=1:length(bias)-1;
                z = weight{i} * activations{i} + repmat(bias{i},1,batchSize);
                zs{i} = z;
                activation = TransferFnc(z);
                activations{i+1} = activation;
            end
            z = weight{length(bias)} * activations{length(bias)} + repmat(bias{length(bias)},1,batchSize);
            zs{length(bias)} = z;
            if strcmp(func2str(TransferFnc),'ReLU')
                activation = Softmax(z);
            else    
                activation = TransferFnc(z);
            end
            activations{length(bias)+1} = activation;
            % When TransferFnc is 'Tanh', output activations are normalized
            % from the range of (-1,1) to (0,1)
            if strcmp(func2str(TransferFnc),'Tanh')
                activations{length(bias)+1} = ( activations{length(bias)+1} + 1 )/2;
            end
        end
        
        % Backward pass
        % Compute output error of the last layer and update last cell of
        % nabla_b and nabla_w
        
        % When CostFunction is 'QuadraticCost', delta is
        % a-y.*TransferPrime(z)
        if strcmp(CostFunction,'QuadraticCost')
            a = activations{length(nodeLayers)};
            y = mini_y{k};
            z = zs{length(bias)};
            delta = (a - y) .* TransferPrime(z);
        end
        
        % When CostFunction is 'CrossEntropyCost', delta is a-y if
        % TransferFnc is Sigmoid and (a-y) ./ ( a.*(1-a) ).* TransferPrime(z)
        % for other TranferFnc
        if strcmp(CostFunction,'CrossEntropyCost')
            a = activations{length(nodeLayers)};
            y = mini_y{k};
            z = zs{length(bias)};
            if strcmp(TransferFnc, 'Sigmoid')
                delta = a-y;
            else
                cost_der = (a-y) ./ ( a.*(1-a) );
                delta = cost_der .* TransferPrime(z);
                delta(isinf(delta) | isnan(delta)) = 0;
            end
        end
        
        % When CostFunction is 'LogLikelihoodCost Fucntion', delta is a-y
        if strcmp(CostFunction,'LogLikelihoodCost')
            a = activations{length(nodeLayers)};
            y = mini_y{k};
            delta = a - y ;
        end 
        
        % Compute last cell of nabla_b & nabla_w
        nabla_b{length(bias)} = sum(delta,2);    
        nabla_w{length(weight)} = delta * transpose( activations{length(nodeLayers)-1} );
                
        % Backpropagate the error and update the second to the last cells
        % through first cells of nabla_b and nabla_w 
        
        % When TransferFnc is 'Softmax', SigmoidPrime is used
        if strcmp(func2str(TransferFnc), 'Softmax')
            for i = 1:length(nodeLayers)-2;
                z = zs{length(bias)-i};
                sp = SigmoidPrime(z);
                delta = ( transpose(weight{length(bias)-i+1}) * delta ) .* sp;
                nabla_b{length(bias)-i} = sum(delta,2);
                nabla_w{length(bias)-i} = delta * transpose( activations{length(nodeLayers)-i-1} );
            end
         % When TransferFnc is other than 'Softmax', given TransferPrime is
         % used
         else
            for i = 1:length(nodeLayers)-2;
                z = zs{length(bias)-i};
                sp = TransferPrime(z);
                delta = ( transpose(weight{length(bias)-i+1}) * delta ) .* sp;
                nabla_b{length(bias)-i} = sum(delta,2);
                nabla_w{length(bias)-i} = delta * transpose( activations{length(nodeLayers)-i-1} );
            end            
         end
        
        % Update weights and biases for one kth mini batch
        % Weight is scaled down with regularization parameter and Momentum 
        % term is added 
        for i=1:length(bias);
             weight{i} = (1-eta*(Reg/size(training_x,2)))*weight{i} - ( (eta/batchSize)*nabla_w{i} + Mom*(eta/batchSize)*prev_nabla{i} );
             bias{i} = bias{i} - (eta/batchSize)*nabla_b{i};
        end
        
        % Delta of n-1 period wich will be used for next momentum 
        % term is reset as nabla_w of current delta
        prev_nabla = nabla_w;
    end
    
    % Training Set
    % Calculate Accuracy and Cost for Training set
    for j=1:size(training_x,2);
        activation = training_x(:,j);
        activations = cell(1,length(nodeLayers));
        activations{1} = training_x(:,j);
        zs = cell(1,length(bias));
               
        if strcmp(func2str(TransferFnc),'Softmax')
            for i=1:length(bias)-1;
                z = weight{i} * activations{i} + bias{i};
                zs{i} = z;
                activation = Sigmoid(z);
                activations{i+1} = activation;
            end
            z = weight{length(bias)} * activations{length(bias)} + bias{length(bias)};
            zs{length(bias)} = z;
            activation = TransferFnc(z);
            activations{length(bias)+1} = activation;
        else
            for i=1:length(bias)-1;
                z = weight{i} * activations{i} + bias{i};
                zs{i} = z;
                activation = TransferFnc(z);
                activations{i+1} = activation;
            end
            z = weight{length(bias)} * activations{length(bias)} + bias{length(bias)};
            zs{length(bias)} = z;
            if strcmp(func2str(TransferFnc),'ReLU')
                activation = Softmax(z);
            else
                activation = TransferFnc(z);
            end
            activations{length(bias)+1} = activation;
            if strcmp(func2str(TransferFnc),'Tanh')
                activations{length(bias)+1} = ( activations{length(bias)+1} + 1 )/2;
            end
        end

        % Training Set - Sum correct classifications 
        if sum( round(activations{length(nodeLayers)}) == training_y(:,j) ) == nodeLayers(end)
            total1 = total1 +1;
        end
        acc1 = total1/size(training_x,2);
        
        % Training Set - QuadraticCost Function Cost Phase I
        if strcmp(CostFunction,'QuadraticCost')
            % Calculate MSE Phase 1 - Sum the squared vector lengths of y(x)-a of each training
            % sample
            cost1 = cost1 + sum( (training_y(:,j)-activations{length(nodeLayers)}).^2 );
        end
        
        % Training Set - CrossEntropyCost Function Cost Phase I
        if strcmp(CostFunction,'CrossEntropyCost')
            y = training_y(:,j);
            a = activations{length(nodeLayers)};
            cost_per = (y.*log(a)+(1-y).*log(1-a));
            cost_per(isinf(cost_per) | isnan(cost_per)) = 0;
            cost1 = cost1 -  sum( cost_per );
        end

        % Training Set - LogLikelihoodCost Function Cost Phase I
        if strcmp(CostFunction,'LogLikelihoodCost')
            y = training_y(:,j);
            position = find(y==1);
            a = activations{length(nodeLayers)};
            cost1 = cost1 -log( a(position) ) ;
        end
    end
    
    % Calculate the regularized term, sum of sqaured weights for L2 Regularization
    reg_term = 0;
    for i=1:length(weight)
        reg_term = reg_term + norm(weight{i})^2;
    end 
    
    % Training Set - QuadraticCost Function Cost Phase II
    if strcmp(CostFunction,'QuadraticCost')          
        cost1 = cost1 / (2*size(training_x,2)) + Reg/(2*size(training_x,2))*reg_term;
    end
    
    % Training Set - CrossEntropyCost Function Cost Phase II
    if strcmp(CostFunction,'CrossEntropyCost')   
        cost1 = cost1 / size(training_x,2) + Reg/(2*size(training_x,2))*reg_term;
    end
    
    % Training Set - LogLikelihoodCost Function Cost Phase II
    if strcmp(CostFunction,'LogLikelihoodCost')    
        cost1 = cost1 / size(training_x,2) + Reg/(2*size(training_x,2))*reg_term;
    end
  
    % Training Set - Print the result for each epoch       
    result1 = '%-3.0f | %-5.3f | %5.0f/%-5.0f | %-3.3f || \b ';
    fprintf(result1,t,cost1,total1,size(training_x,2),acc1); 
    
    % Validation Set
    % Calculate Accuracy and Cost for Validation Set
    for j=1:size(validation_x,2);
        activation = validation_x(:,j);
        activations = cell(1,length(nodeLayers));
        activations{1} = validation_x(:,j);
        zs = cell(1,length(bias));
                
        if strcmp(func2str(TransferFnc),'Softmax')
            for i=1:length(bias)-1;
                z = weight{i} * activations{i} + bias{i};
                zs{i} = z;
                activation = Sigmoid(z);
                activations{i+1} = activation;
            end
            z = weight{length(bias)} * activations{length(bias)} + bias{length(bias)};
            zs{length(bias)} = z;
            activation = TransferFnc(z);
            activations{length(bias)+1} = activation;
        else
            for i=1:length(bias)-1;
                z = weight{i} * activations{i} + bias{i};
                zs{i} = z;
                activation = TransferFnc(z);
                activations{i+1} = activation;
            end
            z = weight{length(bias)} * activations{length(bias)} + bias{length(bias)};
            zs{length(bias)} = z;
            if strcmp(func2str(TransferFnc),'ReLU') 
                activation = Softmax(z);
            else
                activation = TransferFnc(z);
            end
            activations{length(bias)+1} = activation;
            if strcmp(func2str(TransferFnc),'Tanh')
                activations{length(bias)+1} = ( activations{length(bias)+1} + 1 )/2;
            end
        end

        % Validation Set - Sum correct classifications       
        if sum( round(activations{length(nodeLayers)}) == validation_y(:,j) ) == nodeLayers(end)
            total2 = total2 +1;
        end
        acc2 = total2/size(validation_x,2);
        
        % Validation Set - QuadraticCost Function Cost Phase I
        if strcmp(CostFunction,'QuadraticCost')
            cost2 = cost2 + sum( (validation_y(:,j)-activations{length(nodeLayers)}).^2 ) ;
        end
        
        % Validation Set - CrossEntropyCost Function Cost Phase I
        if strcmp(CostFunction,'CrossEntropyCost')
            y = validation_y(:,j);
            a = activations{length(nodeLayers)};
            cost_per = (y.*log(a)+(1-y).*log(1-a));
            cost_per(isinf(cost_per) | isnan(cost_per)) = 0;
            cost2 = cost2 -  sum( cost_per ) ;
        end

        % Validation Set - LogLikelihoodCost Function Cost Phase I
        if strcmp(CostFunction,'LogLikelihoodCost')
            y = validation_y(:,j);
            position = find(y==1);
            a = activations{length(nodeLayers)};
            cost2 = cost2 -log( a(position) ) ;
        end
    end
    
    % Validation Set - QuadraticCost Function Cost PhaseII
    if strcmp(CostFunction,'QuadraticCost')   
        cost2 = cost2 / (2*size(validation_x,2) ) + Reg/(2*size(validation_x,2))*reg_term;
    end
    
    % Validation Set - CrossEntropyCost Function Cost Phase II
    if strcmp(CostFunction,'CrossEntropyCost')
        % Calculate MSE Phase 2 - Divide sum of squared vector lengths by 2*number of samples    
        cost2 = cost2 / size(validation_x,2) + Reg/(2*size(validation_x,2))*reg_term;
    end
    
    % Validation Set - LogLikelihoodCost Function Cost Phase II
    if strcmp(CostFunction,'LogLikelihoodCost')
        % Calculate MSE Phase 2 - Divide sum of squared vector lengths by 2*number of samples    
        cost2 = cost2 / size(validation_x,2) + Reg/(2*size(validation_x,2))*reg_term;
    end
    
    % Validation Set - Print the result for each epoch      
    result2 = ' %-5.3f | %5.0f/%-5.0f | %-3.3f || \b ';
    fprintf(result2,cost2,total2,size(validation_x,2),acc2); 

    % Test Set
    % Calculate Accuracy and Cost for Test Set
    for j=1:size(test_x,2);
        activation = test_x(:,j);
        activations = cell(1,length(nodeLayers));
        activations{1} = test_x(:,j);
        zs = cell(1,length(bias));
                
        if strcmp(func2str(TransferFnc),'Softmax')
            for i=1:length(bias)-1;
                z = weight{i} * activations{i} + bias{i};
                zs{i} = z;
                activation = Sigmoid(z);
                activations{i+1} = activation;
            end
            z = weight{length(bias)} * activations{length(bias)} + bias{length(bias)};
            zs{length(bias)} = z;
            activation = TransferFnc(z);
            activations{length(bias)+1} = activation;
        else
            for i=1:length(bias)-1;
                z = weight{i} * activations{i} + bias{i};
                zs{i} = z;
                activation = TransferFnc(z);
                activations{i+1} = activation;
            end
            z = weight{length(bias)} * activations{length(bias)} + bias{length(bias)};
            zs{length(bias)} = z;
            if strcmp(func2str(TransferFnc),'ReLU') 
                activation = Softmax(z);
            else
                activation = TransferFnc(z);
            end
            activations{length(bias)+1} = activation;
            if strcmp(func2str(TransferFnc),'Tanh')
                activations{length(bias)+1} = ( activations{length(bias)+1} + 1 )/2;
            end
        end
    
        % Test Set - Sum correct classifications       
        if sum( round(activations{length(nodeLayers)}) == test_y(:,j) ) == nodeLayers(end)
            total3 = total3 +1;
        end
        acc3 = total3/size(test_x,2);
        
        % Test Set - QuadraticCost Function Cost Phase I: Sum the squared vector lengths of y(x)-a 
        % of each training sample
        if strcmp(CostFunction,'QuadraticCost')
            cost3 = cost3 + sum( (test_y(:,j)-activations{length(nodeLayers)}).^2 ) ;
        end
        
        % Test Set - CrossEntropyCost Function Cost Phase I
        if strcmp(CostFunction,'CrossEntropyCost')
            y = test_y(:,j);
            a = activations{length(nodeLayers)};
            cost_per = (y.*log(a)+(1-y).*log(1-a));
            cost_per(isinf(cost_per) | isnan(cost_per)) = 0;
            cost3 = cost3 -  sum( cost_per ) ;
        end

        % Test Set - LogLikelihoodCost Function Cost Phase I
        if strcmp(CostFunction,'LogLikelihoodCost')
            y = test_y(:,j);
            position = find(y==1);
            a = activations{length(nodeLayers)};
            cost3 = cost3 -log( a(position) ) ;
        end
    end
    
    % Test Set - QuadraticCost Function Cost Phase II: Divide sum of 
    % squared vector lengths by 2*number of samples
    if strcmp(CostFunction,'QuadraticCost')  
        cost3 = cost3 / (2*size(test_x,2)) + Reg/(2*size(test_x,2))*reg_term;
    end
    
    % Test Set - CrossEntropyCost Function Cost Phase II
    if strcmp(CostFunction,'CrossEntropyCost')   
        cost3 = cost3 / size(test_x,2) + Reg/(2*size(test_x,2))*reg_term;
    end
    
    % Test Set - LogLikelihoodCost Function Cost Phase II
    if strcmp(CostFunction,'LogLikelihoodCost') 
        cost3 = cost3 / size(test_x,2) + Reg/(2*size(test_x,2))*reg_term;
    end
    
    % Test Set - Print the result for each epoch
    result3 = ' %-5.3f | %5.0f/%-5.0f | %-3.3f \n ';
    fprintf(result3,cost3,total3,size(test_x,2),acc3);
    
    % Make a list of accuracies & costs of each epoch for plotting
    acc_train(t) = acc1;
    acc_val(t) = acc2;
    acc_test(t) = acc3;
    cost_train(t) = cost1;
    cost_val(t) = cost2;
    cost_test(t) = cost3;
    
    % If EarlyStop is True, Indicator turns to 'False' if there is no
    % improvement within 10 epochs and while loop ends
    if strcmp(EarlyStop,'True')
        if t > 10
            cnt = 0;
            for i=0:9
                if acc_val(end-i) - acc_val(end-i-1) <= 0
                    cnt = cnt + 1;
                end
            end
            if cnt == 10
                indicator = 'False';
            end
        end
    end            
            
    end
    
    % Plot accuracies & costs over epochs
    x = 1:t;
    figure
    plot(x,acc_train,x,acc_val,x,acc_test)
    legend('Train','Val','Test','Location','Northeastoutside')
    title('Accuracy over Epochs')
    xlabel('Number of Epochs')
    ylabel('Accuracy')
    figure
    plot(x,cost_train,x,cost_val,x,cost_test)
    legend('Train','Val','Test','Location','Northeastoutside')
    title('Cost over Epochs')
    xlabel('Number of Epochs')
    ylabel('Cost')
    
    % Assign weight & bias as output learned_weight & learned_bias
    learned_weight = weight;
    learned_bias = bias;
end
 
