%%
% Compute features for a set of video files from datasets
% 
close all; 
clear;

% add path
addpath(genpath('include'));

%%
% parameters
algo_name = 'GAMIVAL'; % algorithm name, eg, 'V-BLIINDS'
data_name = 'GamingVideoSET';  % dataset name, eg, 'KONVID_1K'
write_file = true;  % if true, save features on-the-fly
log_level = 0;  % 1=verbose, 0=quite

if strcmp(data_name, 'YT-UGC-Gaming')
    data_path = '/media/LIVELAB/berrie/FB_Project/YouTube_Gaming/original_videos';
elseif strcmp(data_name, 'LIVE-YT-Gaming')
    data_path = '/Volumes/Seagate/Dataset/LIVE-YT-Gaming';
elseif strcmp(data_name, 'GamingVideoSET')
    data_path = '/Volumes/Seagate/Dataset/GamingVideoSET';
elseif strcmp(data_name, 'KUGVD')
    data_path = '/Volumes/Seagate/Dataset/KUGVD';
elseif strcmp(data_name, 'LIVE-Meta-Gaming')
    data_path = '/Volumes/Seagate/Dataset/LIVE-Meta-Gaming';
elseif strcmp(data_name, 'CGVDS')
    data_path = '/Volumes/Seagate/Dataset/CGVDS';
end

%%
% create temp dir to store decoded videos
video_tmp = fullfile(data_path, 'YUV');
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = 'mos_files';
filelist_csv = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(filelist_csv);
num_videos = size(filelist,1);
out_path = 'feat_files';
if ~exist(out_path, 'dir'), mkdir(out_path); end
out_mat_name = fullfile(out_path, [data_name,'_',algo_name,'_feats.mat']);
out_mat_frame_name = fullfile(out_path, [data_name,'_',algo_name,'_feats_frame.mat']);
feats_mat = [];
%===================================================

%% extract features
% parfor i = 1:num_videos % for parallel speedup
for i = 1:num_videos
    progressbar(i/num_videos) % Update figure
    if strcmp(data_name, 'YT-UGC-Gaming')
        video_name = fullfile(data_path, ...
            [num2str(filelist.resolution(i)),'P'],[filelist.vid{i},'.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    elseif strcmp(data_name, 'LIVE-YT-Gaming')
        video_name = fullfile(data_path, filelist.File{i});
        yuv_name = fullfile(video_tmp, [filelist.File{i}, '.yuv']);
    elseif strcmp(data_name, 'GamingVideoSET')
        video_name = fullfile(data_path, filelist.File{i});
        yuv_name = fullfile(video_tmp, [filelist.File{i}, '.yuv']);
    elseif strcmp(data_name, 'KUGVD')
        video_name = fullfile(data_path, filelist.File{i});
        yuv_name = fullfile(video_tmp, [filelist.File{i}, '.yuv']);
    elseif strcmp(data_name, 'LIVE-Meta-Gaming')
        video_name = fullfile(data_path, filelist.File{i});
        yuv_name = fullfile(video_tmp, [filelist.File{i}, '.yuv']);
    elseif strcmp(data_name, 'CGVDS')
        video_name = fullfile(data_path, filelist.File{i});
        yuv_name = fullfile(video_tmp, [filelist.File{i}, '.yuv']);
    end
    fprintf('\n\nComputing features for %d sequence: %s\n', i, video_name);

    % decode video and store in temp dir
    cmd = ['ffmpeg -loglevel error -y -i ', video_name, ...
        ' -pix_fmt yuv420p -vsync 0 ', yuv_name];
    system(cmd);  

    % get video meta data
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));

    % calculate video features
    tStart = tic;
    feats_frames = calc_GAMIVAL_features(yuv_name, width, height, ...
        framerate, log_level);
    fprintf('\nOverall %f seconds elapsed...', toc(tStart));
    feats_mat(i,:) = nanmean(feats_frames);
    % clear cache
    delete(yuv_name)

    if write_file
        save(out_mat_name, 'feats_mat');
    end
end
