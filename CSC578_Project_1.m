% CSC 578 Project 1 Vicky Lee
function report = CSC578_Project_1(inputs, targets, nodeLayers, numEpochs, batchSize, eta)
 
% Initialize weights and biases with random numbers of normal distribution
% with mean 0 and standard deviation 1

bias = cell(1,length(nodeLayers)-1);
for i = 1:length(nodeLayers)-1;
    bias{i} = randn(nodeLayers(i+1),1);
end
 
weight = cell(1,length(nodeLayers)-1);
for i = 1:length(nodeLayers)-1;
    weight{i} = randn(nodeLayers(i+1),nodeLayers(i));
end

t = 0;
total = 0;
% Repeat epoch while iteration is less than or equal to numEpochs and 
% not all samples are correctly classified
while (t <= numEpochs-1 & total < size(targets,2)) 
    
    % Create variables t(iteration number for numEpochs), total(correctly
    % classified samples, and cost(MSE)
    t = t+1;
    total = 0;
    cost = 0;
    
    % Shuffle the training data 
    order = randperm(size(inputs,2));
    inputs(:,order);
    targets(:,order);
 
    % Split the training data, inputs and targets into mini batches
    % by batchSize
    mini_x = cell(1,size(inputs,2)/batchSize);
    mini_y = cell(1,size(inputs,2)/batchSize);
    n=0;
    for i=1:batchSize:size(inputs,2)-batchSize+1;
        n = n+ 1;
        mini_x{n} = inputs(:,i:i+batchSize-1);
        mini_y{n} = targets(:,i:i+batchSize-1);
    end
    
    % Mini-Batch SGD
    % Update weights and biases through all mini batches
    for k=1:length(mini_x);
     
        nabla_b = cell(1,length(bias));
        nabla_w = cell(1,length(weight));
            
        % feedforward       
        activation = mini_x{k};
        activations = cell(1,length(nodeLayers));
        activations{1} = mini_x{k};
        zs = cell(1,length(bias));
    
        for i=1:length(bias);
            z = weight{i} * activations{i} + repmat(bias{i},1,batchSize);
            zs{i} = z;
            activation = logsig(z);
            activations{i+1} = activation;
        end
        
        % Sum correct classifications       
        mini_total = sum( sum( round(activations{length(nodeLayers)}) == mini_y{k} ) == nodeLayers(end) );
        total = total + mini_total;
        
        % Calculate MSE Phase 1 - Sum the squared vector lengths of y(x)-a of each training
        % sample
        mini_cost = 0;
        for j=1:batchSize;
            mini_cost = mini_cost + (norm( mini_y{k}(:,j)- activations{length(nodeLayers)}(:,j) ) )^2;
        end
        cost = cost + mini_cost;
 
        % Backward pass
        % Compute output error of the last layer and update last cell of
        % nabla_b and nabla_w
        delta = (activations{length(nodeLayers)} - mini_y{k}) .* ( logsig(zs{length(bias)}) .* (1-logsig(zs{length(bias)}) ));
        nabla_b{length(bias)} = sum(delta,2);    
        nabla_w{length(weight)} = delta * transpose( activations{length(nodeLayers)-1} );
        
        % Backpropagate the error and update the second to the last cells
        % through first cells of nabla_b and nabla_w
        for i = 1:length(nodeLayers)-2;
            z = zs{length(bias)-i};
            sp = logsig(z).*(1-logsig(z));
            delta = ( transpose(weight{length(bias)-i+1}) * delta ) .* sp;
            nabla_b{length(bias)-i} = sum(delta,2);
            nabla_w{length(bias)-i} = delta * transpose( activations{length(nodeLayers)-i-1} );
        end
 
        % Update weights and biases for one kth mini batch
        for i=1:length(bias);
             weight{i} = weight{i} - (eta/batchSize)*nabla_w{i};
             bias{i} = bias{i} - (eta/batchSize)*nabla_b{i};
        end
        
        end
    
    % Calculate MSE Phase 2 - Divide sum of squared vector lengths by 2*number of samples    
    cost = cost / (2*size(inputs,2));
    
    % Print the result for each epoch
    result = 'Epoch %3.0f, MSE: %5.4f, Correct: %5.0f / %5.0f, Acc: %3.2f \n';
    fprintf(result,t,cost,total,size(inputs,2),total/size(inputs,2)); 
    end
end
 
