%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = 100;
k = 10;
bNoise = 1;
idx = 1;
cr = 0.1;
    
%data_file = strcat('./data/RL_data_nn_', num2str(n_o), '.mat');
%data_file = strcat('./data/', num2str(n_o), '.mat');
%data_file = strcat('D:/Dataset/RLHH/', num2str(k), 'K_', 'p', num2str(p), '_', noise_str, num2str(n_o), '_', num2str(idx), '.mat');
data_file = FindDataPath( p, k, cr, bNoise, idx );
data = load(data_file);
Xtr = data.Xtr;
ytr = data.ytr;
beta_truth = data.beta;

%% Test different data sets
tic;
[beta, S] = RACT(Xtr, ytr);
toc;
beta_truth_norm = norm(beta_truth);
beta_norm = norm(beta);

fprintf('[%dK|p%d|%.2f] - |w-w*|: %f\n', k, p, cr, norm(beta_truth-beta));
