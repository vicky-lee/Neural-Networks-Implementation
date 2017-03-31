function dsoft = SoftmaxPrime(x)
  soft = Softmax(x);
  dsoft = soft .* (1-soft);
  
end
