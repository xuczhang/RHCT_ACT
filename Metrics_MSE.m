function mse = Metrics_MSE(Xte, Yte, beta)
    %METRICS_AVGERROR Summary of this function goes here
    %   Detailed explanation goes here
    
    Yte_est = Xte'*beta;
    
    mse = sqrt(sum(abs(Yte_est - Yte))/(size(Xte, 2)));
end
