%% =========================================================================
% Compute video features for a dataset using a specified VQA algorithm
% =========================================================================

% Close all figures and clear workspace
close all; 
clear;

% Add necessary folders to path (recursively)
addpath(genpath('include'));

%% =========================================================================
% Parameters
% =========================================================================

algo_name = 'GAMIVAL';              % Algorithm name, e.g., 'V-BLIINDS', 'GAMIVAL'
data_name = 'GamingVideoSET';       % Dataset name, e.g., 'KONVID_1K'
write_file = true;                  % If true, saves features while processing
log_level = 0;                      % Log verbosity (1 = verbose, 0 = quiet)

% Set dataset path based on data_name
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

%% =========================================================================
% Prepare directories and read metadata
% =========================================================================

% Temporary directory for decoded YUV videos
video_tmp = fullfile(data_path, 'YUV');
if ~exist(video_tmp, 'dir')
    mkdir(video_tmp);
end

% Load video metadata
feat_path = 'mos_files';
filelist_csv = fullfile(feat_path, [data_name, '_metadata.csv']);
filelist = readtable(filelist_csv);
num_videos = size(filelist, 1);

% Output path for storing features
out_path = 'feat_files';
if ~exist(out_path, 'dir')
    mkdir(out_path);
end

% Output filenames
out_mat_name = fullfile(out_path, [data_name, '_', algo_name, '_feats.mat']);
out_mat_frame_name = fullfile(out_path, [data_name, '_', algo_name, '_feats_frame.mat']);

% Initialize feature matrix
feats_mat = [];

%% =========================================================================
% Main feature extraction loop
% =========================================================================

% Use `parfor` here for parallel processing if needed
for i = 1:num_videos
    progressbar(i / num_videos);  % Display progress bar in figure window

    % Construct video and YUV file names based on dataset type
    if strcmp(data_name, 'YT-UGC-Gaming')
        video_name = fullfile(data_path, ...
            [num2str(filelist.resolution(i)), 'P'], [filelist.vid{i}, '.mkv']);
        yuv_name = fullfile(video_tmp, [filelist.vid{i}, '.yuv']);
    else
        video_name = fullfile(data_path, filelist.File{i});
        yuv_name = fullfile(video_tmp, [filelist.File{i}, '.yuv']);
    end

    fprintf('\n\nComputing features for %d sequence: %s\n', i, video_name);

    % Use FFmpeg to decode video to YUV format (YUV420p)
    cmd = ['ffmpeg -loglevel error -y -i ', video_name, ...
           ' -pix_fmt yuv420p -vsync 0 ', yuv_name];
    system(cmd);

    % Read video resolution
    % For most datasets, use original width and height from metadata
    % For LIVE-Meta-Gaming, use DisplayWidth and DisplayHeight instead of TrueWidth and TrueHeight,
    % since the videos should be upscaled using bicubic interpolation before feature extraction
    if strcmp(data_name, 'LIVE-Meta-Gaming')
        width = filelist.DisplayWidth(i);
        height = filelist.DisplayHeight(i);

        % NOTE: Make sure that all videos in the LIVE-Meta-Gaming database
        % are upscaled from (TrueWidth, TrueHeight) to (DisplayWidth, DisplayHeight)
        % using bicubic interpolation before feature extraction.
        %
        % This ensures consistency with the subjective testing conditions.
    else
        width = filelist.width(i);
        height = filelist.height(i);
    end
    framerate = round(filelist.framerate(i));

    % Extract features using the GAMIVAL algorithm
    tStart = tic;
    feats_frames = calc_GAMIVAL_features(yuv_name, width, height, ...
                                         framerate, log_level);
    fprintf('\nOverall %f seconds elapsed...', toc(tStart));

    % Average frame-level features to get video-level representation
    feats_mat(i, :) = nanmean(feats_frames);

    % Delete temporary YUV file
    delete(yuv_name);

    % Save intermediate results to MAT file
    if write_file
        save(out_mat_name, 'feats_mat');
    end
end
