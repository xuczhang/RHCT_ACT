function [] = DataSampling_large(p, k, cr, bNoise, idx)
    
    data_file = FindDataPath( p, k, cr, bNoise, idx );
    fprintf('=== %s ===\n', data_file);
    
    bs = 10; % batch_size = 10K
    bn = k / bs; % number of batches
    
    beta = DataSampling_beta(p);
    
    Xtr = [];
    ytr = [];
    S = [];
    for i = 1:bn
        %fprintf('=== %g ===\n', i);
        [Xtr_i, ytr_i] = DataSampling_batch( beta, p, bs, cr, bNoise);
        Xtr = [Xtr Xtr_i];
        ytr = [ytr; ytr_i];
        S = [S (i-1)*bs*1000 + 1: (i-1)*bs*1000 + bs*1000*(1-cr)];
    end
    
    save(data_file, 'Xtr', 'ytr', 'beta', 'S');
end

function beta = DataSampling_beta(p)
    %% Generate the training sample data
    % sample w by unit norm vector in p dimension
    beta = rand(p, 1);
    beta_norm = norm(beta);
    beta = beta/beta_norm;
end

function [Xtr, ytr, S] = DataSampling_batch( beta, p, k, cr, bNoise)

    %% Initialize the constant
    %p = 100; % feature dimension
    %k = 1;
    %cr = 0.1; % corruption ratio (from 0.1 to 1.2)
    %bNoise = 1;

    n = 1000*k; % total sample number in training data
    n_o = int32(cr*n); % corruption sample number(from 100 to 1200)
    n_u = n - n_o;

    % sample x by normal distribution with mu=0 and cov=I_p
    X_mu = zeros(p, 1);
    X_cov = eye(p);
    Xtr_a = mvnrnd(X_mu, X_cov, n_u)'; % authetic data
    Xtr_o = mvnrnd(X_mu, X_cov, n_o)'; % outlier part X
    Xtr = [Xtr_a, Xtr_o];  
    
    % sample noise eplison for outliers
    e_mu = zeros(n_u, 1);
    e_cov = eye(n_u) * 0.1;
    e_a = mvnrnd(e_mu, e_cov)';

    % generate the authentic samples by y_i = <w, x_i> + v_i
    if bNoise
        ytr_a = Xtr_a'*beta + e_a;
    else
        ytr_a = Xtr_a'*beta;
    end

    %% Generate Training Outlier Data
    % sample corruption vector b as b_i ~ U(-5|y*|_inf, 5|y*|_inf)
    u_range = 5*norm(ytr_a, inf);
    u = -u_range + 2*u_range*rand(n_o,1);

    % sample noise eplison for outliers
    e_mu = zeros(n_o, 1);
    e_cov = eye(n_o) * 0.01;
    e_o = mvnrnd(e_mu, e_cov)';

    % generate outlier y: ytr_o = sign(<-beta, x_o>)
    if bNoise
        ytr_o = Xtr_o'*beta + u + e_o;
    else
        ytr_o = Xtr_o'*beta + u;
    end

    ytr = [ytr_a; ytr_o];

end

