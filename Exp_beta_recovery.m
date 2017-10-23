p = 200;
k = 20;
bNoise = 1;

n = 1000*k;
dup_num = 10;

OLS_result = [];
DALM_result = [];
HOMO_result = [];
TORRENT50_result = [];
TORRENT25_result = [];
TORRENT0_result = [];
RLHH_result = [];
RACT_result = [];

fprintf('************ p=%d k=%d bNoise=%d ************\n', p, k, bNoise);

for cr = 0.05:0.05:0.4

    if bNoise == 1
        noise_str = ''; 
    else
        noise_str = 'nn_';
    end
    n_o = int32(cr*n);
    
    OLS_err = 0;
    DALM_err = 0;
    HOMO_err = 0;
    TORRENT0_err = 0;
    TORRENT25_err = 0;
    TORRENT50_err = 0;
    RLHH_err = 0;
    RACT_err = 0;
    
        
    for idx = 1:1:dup_num
            
        %data_file = strcat('D:/Dataset/RLHH/', num2str(k), 'K_', 'p', num2str(p), '_', noise_str, num2str(n_o), '_', num2str(idx), '.mat');
        data_file = FindDataPath( p, k, cr, bNoise, idx );
        data = load(data_file);
        Xtr = data.Xtr;
        ytr = data.ytr;
        beta_truth = data.beta;

        %% Test different methods
        fprintf('=== [%f] / %d ===\n', cr, idx);

        
%         % Ordinary Least Square
%         OLS_beta = regress(ytr, Xtr');
%         OLS_err = OLS_err + norm(beta_truth-OLS_beta);        
% 
        % DALM Method
        STOPPING_TIME = -2 ;
        maxTime = 8;
        DALM_beta = Baseline_DALM_CBM(Xtr', ytr, 'stoppingCriterion', STOPPING_TIME, 'groundtruth', beta_truth, 'maxtime', maxTime, 'maxiteration', 1e6);
        DALM_err = DALM_err + norm(beta_truth-DALM_beta);

        % Homotopy Method
        HOMO_beta = Baseline_Homotopy_CBM(Xtr', ytr, 'stoppingCriterion', STOPPING_TIME, 'groundtruth', beta_truth, 'maxtime', maxTime, 'maxiteration', 1e6);
        HOMO_err = HOMO_err + norm(beta_truth-HOMO_beta);
        
        
        % TORRENT0
        TORRENT0_beta = Baseline_TORRENT( Xtr, ytr, cr);
        TORRENT0_err = TORRENT0_err + norm(beta_truth-TORRENT0_beta); 
        
        % TORRENT25
        total_error = 0;
        for ecr = 0.75*cr:0.5/20*cr:1.25*cr
            TORRENT25_beta = Baseline_TORRENT( Xtr, ytr, ecr);
            total_error = total_error + norm(beta_truth-TORRENT25_beta);
        end    
        TORRENT25_err = TORRENT25_err + total_error/21;
        
        % TORRENT50
        total_error = 0;
        for ecr = 0.5*cr:1/20*cr:1.5*cr
            TORRENT50_beta = Baseline_TORRENT( Xtr, ytr, ecr);
            total_error = total_error + norm(beta_truth-TORRENT50_beta);
        end    
        TORRENT50_err = TORRENT50_err + total_error/21;

        % RLHH
        [RLHH_beta, S] = RLHH(Xtr, ytr);
        RLHH_err = RLHH_err + norm(beta_truth-RLHH_beta);     
        
        % RACT
        [RACT_beta, S] = RACT(Xtr, ytr);
        RACT_err = RACT_err + norm(beta_truth-RACT_beta); 
    
    end
    OLS_result = [OLS_result OLS_err/dup_num];
    DALM_result = [DALM_result DALM_err/dup_num];
    HOMO_result = [HOMO_result HOMO_err/dup_num];
    TORRENT0_result = [TORRENT0_result TORRENT0_err/dup_num];
    TORRENT25_result = [TORRENT25_result TORRENT25_err/dup_num];    
    TORRENT50_result = [TORRENT50_result TORRENT50_err/dup_num];    
    RLHH_result = [RLHH_result RLHH_err/dup_num];
    RACT_result = [RACT_result RACT_err/dup_num];
    
    %fprintf('[%d] - |w-w*|: %f outlier_idx:%d \n', n_o, total_error/21, size(S, 1));
    %fprintf('[%d] - |w-w*|: %f outlier_idx:%d \n', n_o, norm(w_truth-TORRENT_w), size(S, 1));
end
%result_path = 'D:/Dropbox/PHD/publications/IJCAI2017_RLHH/experiment/';
result_path = FindResultPath();
file_output = strcat(result_path, 'beta_', num2str(k), 'K_', 'p', num2str(p), '_', noise_str);
file_output = file_output(1:end-1);
save(file_output, 'OLS_result', 'DALM_result', 'HOMO_result', 'TORRENT0_result', 'TORRENT25_result', 'TORRENT50_result', 'RLHH_result', 'RACT_result');
