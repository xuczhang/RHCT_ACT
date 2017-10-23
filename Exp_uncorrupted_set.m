p = 400;
k = 100;
cr = 0.4;
bNoise = 0;

n = 1000*k;
dup_num = 10;

if bNoise == 1
    noise_str = ''; 
else
    noise_str = 'nn_';
end

n_o = int32(cr*n);
n_u = n - n_o;

TORRENT0_f1 = 0;
TORRENT25_f1 = 0;
TORRENT50_f1 = 0;
TORRENT80_f1 = 0;
RLHH_f1 = 0;
RACT_f1 = 0;

%S_truth = [ones(n_u, 1) ; zeros(n_o, 1)];
S_truth = zeros(n, 1);

fprintf('=== cr:%f ===\n', cr);
for idx = 1:1:dup_num

    data_file = FindDataPath( p, k, cr, bNoise, idx );
    data = load(data_file);
    Xtr = data.Xtr;
    ytr = data.ytr;
    w_truth = data.beta;
    S = data.S;
    S_truth(S) = 1;
    
    % TORRENT0
    [TORRENT0_w, TORRENT0_S] = Baseline_TORRENT( Xtr, ytr, cr);
    S_TORR0 = zeros(n_u+n_o, 1);
    S_TORR0(TORRENT0_S) = 1;
    stat_TORR0 = confusionmatStats(S_truth, S_TORR0);
    TORRENT0_f1 = TORRENT0_f1 + stat_TORR0.Fscore(2);
    
    % TORRENT80
    [TORRENT80_w, TORRENT80_S] = Baseline_TORRENT( Xtr, ytr, 1.8*cr);
    S_TORR80 = zeros(n_u+n_o, 1);
    S_TORR80(TORRENT80_S) = 1;
    stat_TORR80 = confusionmatStats(S_truth, S_TORR80);
    TORRENT80_f1 = TORRENT80_f1 + stat_TORR80.Fscore(2);    
    
    % TORRENT50
    [TORRENT50_w, TORRENT50_S] = Baseline_TORRENT( Xtr, ytr, 1.5*cr);
    S_TORR50 = zeros(n_u+n_o, 1);
    S_TORR50(TORRENT50_S) = 1;
    stat_TORR50 = confusionmatStats(S_truth, S_TORR50);
    TORRENT50_f1 = TORRENT50_f1 + stat_TORR50.Fscore(2);
    
    % TORRENT25
    [TORRENT25_w, TORRENT25_S] = Baseline_TORRENT( Xtr, ytr, 1.25*cr);
    S_TORR25 = zeros(n_u+n_o, 1);
    S_TORR25(TORRENT25_S) = 1;
    stat_TORR25 = confusionmatStats(S_truth, S_TORR25);
    TORRENT25_f1 = TORRENT25_f1 + stat_TORR25.Fscore(2);
    
    % RLHH
    [RLHH_w, RLHH_S] = RLHH(Xtr, ytr);
    S_RLHH = zeros(n_u+n_o, 1);
    S_RLHH(RLHH_S) = 1;
    stat_RLHH = confusionmatStats(S_truth, S_RLHH);
    RLHH_f1 = RLHH_f1 + stat_RLHH.Fscore(2);
    
    % RACT
    [RACT_w, RACT_S] = RACT(Xtr, ytr);
    S_RACT = zeros(n_u+n_o, 1);
    S_RACT(RACT_S) = 1;
    stat_RACT = confusionmatStats(S_truth, S_RACT);
    RACT_f1 = RACT_f1 + stat_RACT.Fscore(2);

end

fprintf('\nTORR80: %.3f\n', TORRENT80_f1/dup_num);
fprintf('TORR50: %.3f\n', TORRENT50_f1/dup_num);
fprintf('TORR25: %.3f\n', TORRENT25_f1/dup_num);
fprintf('RLHH: %.3f\n', RLHH_f1/dup_num);
fprintf('RACT: %.3f\n', RACT_f1/dup_num);
fprintf('TORR0: %.3f\n', TORRENT0_f1/dup_num);