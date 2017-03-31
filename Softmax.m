function soft = Softmax(x)
    soft = exp(x);
    sum_layer = sum(soft,1);
    for i = 1:length(sum_layer)
        soft(:,i) = soft(:,i)/sum_layer(i);
    end
end

