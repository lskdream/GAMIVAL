data_name = 'LIVE-Meta-Mobile-Cloud-Gaming';

load(['feat_files/', data_name, '_CNN_bicubic_feats.mat'])
f = feats_mat;

load(['feat_files/', data_name, '_NSS_bicubic_feats.mat'])

feats_mat = [feats_mat f];

save(['feat_files/', data_name, '_GAMIVAL_feats.mat'], 'feats_mat');
