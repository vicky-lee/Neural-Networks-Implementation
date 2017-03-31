function dlu  = ReLU(x)
    if x > 0
        dlu = 1;
    else
        dlu = 0;
    end
end
