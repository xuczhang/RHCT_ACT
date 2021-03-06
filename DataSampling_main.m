

% generate the single data
p = 100;
k = 100;
bNoise = 1;
idx = 1;
cr = 0.3;
DataSampling_large( p, k, cr, bNoise, idx);

% generate the data per different corruption ratio
% p = 200; % feature dimension
% k = 20;
% bNoise = 1;
% for cr = 0.05:0.05:0.4
%     for idx = 1:1:10
%         DataSampling_large( p, k, cr, bNoise, idx);
%         %DataSampling( p, k, cr, bNoise, idx);
%     end
%     
% end

% generate the data per different feature number
% k = 1;
% cr = 0.1;
% bNoise = 1;
% for p = 50:50:500
%     for idx = 1:1:10
%         %DataSampling_large( p, k, cr, bNoise, idx);
%         DataSampling( p, k, cr, bNoise, idx);
%     end
%     
% end

% generate the data per different data size
% p = 100;
% cr = 0.1;
% bNoise = 1;
% for k = 1:1:10
%     for idx = 1:1:5
%         DataSampling( p, k, cr, bNoise, idx);
%     end
% end

% generate the large data per different data size
% p = 100;
% cr = 0.1;
% bNoise = 1;
% for k = 100:100:900
%     for idx = 1:1:5
%         DataSampling_large( p, k, cr, bNoise, idx);
%     end
% end