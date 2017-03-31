function dtan = TanhPrime(x)
  tan = Tanh(x);
  dtan = 1 - tan.^2;
end
