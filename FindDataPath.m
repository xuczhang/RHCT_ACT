function [ data_file ] = FindDataPath( p, k, cr, bNoise, idx )
%MAPDATAPATH Summary of this function goes here
%   Detailed explanation goes here
    
    %data_path = '~/Dataset/PersonPred/synthetic/';
    data_path = '../../../../../../Dataset/RLHH/';    
    n_o = int32(cr*k*1000);
    
    str_noise = '';
    if ~bNoise
        str_noise = 'nn_';
    end
    data_file = strcat(data_path, num2str(k), 'K_', 'p', num2str(p), '_', str_noise, num2str(n_o), '_', num2str(idx), '.mat');
    
end

