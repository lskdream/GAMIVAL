function feats_frames = calc_GAMIVAL_features(test_video, width, height, ...
                                            framerate, log_level)
    feats_frames = [];
    % Try to open test_video; if cannot, return
    test_file = fopen(test_video,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        return;
    end
    % Open test video file
    fseek(test_file, 0, 1);
    file_length = ftell(test_file);
    if log_level == 1
        fprintf('Video file size: %d bytes (%d frames)\n',file_length, ...
                floor(file_length/width/height/1.5));
    end
    % get frame number
    nb_frames = floor(file_length/width/height/1.5);
    
    % get features for each chunk
    for fr = floor(framerate/2):framerate:nb_frames-2
        
        % read uniformly sampled 3 frames for each 1-sec chunk
        prev_YUV_frame = YUVread(test_file,[width height],max(1,fr-floor(framerate/3)));
        next_YUV_frame = YUVread(test_file,[width height],min(nb_frames-2,fr+floor(framerate/3)));
    	prev_rgb = ycbcr2rgb(uint8(prev_YUV_frame));
        next_rgb = ycbcr2rgb(uint8(next_YUV_frame));
        
        % subsample to 540p resolution
        sside = min(size(prev_YUV_frame,1), size(prev_YUV_frame,2));
        ratio = 540 / sside;
        if ratio < 1
            prev_rgb = imresize(prev_rgb, ratio);
            next_rgb = imresize(next_rgb, ratio);
        end
        
        feats_per_frame = [];
        
        %% extract spatial NSS features - 680-dim
        if log_level == 1
        tic
        fprintf('- Extracting Spatial NSS features (2 fps) ...')
        end
        prev_feats_spt = GAMIVAL_spatial_features(prev_rgb);
        next_feats_spt = GAMIVAL_spatial_features(next_rgb);
        if log_level == 1, toc; end
        
        %% mean pooling of spatial features within chunk
        feats_spt_mean = nanmean([prev_feats_spt; next_feats_spt]);
        feats_per_frame = [feats_per_frame, feats_spt_mean];
        
        %% extract temporal NSS features - 476-dim
        if log_level == 1
        fprintf('- Extracting temporal NSS features (8 fps) ...')
        tic
        end
        wfun = load(fullfile('include', 'WPT_Filters', 'haar_wpt_3.mat'));
        wfun = wfun.wfun;
        frames_wpt = zeros(size(prev_rgb, 1), size(prev_rgb, 2), size(wfun, 2));
        fr_idx_start = max(1, fr - floor(size(wfun, 2) / 2));
        fr_idx_end = min(nb_frames - 3, fr_idx_start + size(wfun, 2) - 1);
        fr_wpt_cnt = 1;
        % read enough number of frames for temporal bandpass
        for fr_wpt = fr_idx_start:fr_idx_end
            YUV_tmp = YUVread(test_file, [width height], fr_wpt);
            if ratio < 1
                frames_wpt(:,:,fr_wpt_cnt) = imresize(YUV_tmp(:,:,1), ratio);
            else
                frames_wpt(:,:,fr_wpt_cnt) = YUV_tmp(:,:,1);
            end
            fr_wpt_cnt = fr_wpt_cnt + 1;
        end
        dpt_filt_frames = zeros(size(prev_rgb, 1), size(prev_rgb, 2), size(wfun, 1));
        feats_tmp_wpt = [];
        w=1.5;
        % compute 1D convolution along time (3rd) dimension
        for freq = 1:size(wfun, 1)
            dpt_filt_frames(:,:,freq) = sum(frames_wpt .* ...
                reshape(wfun(freq,:),1,1,[]), 3);
            dpt_filt_frames(:,:,freq) = dpt_filt_frames(:,:,freq) + randn(size(dpt_filt_frames(:,:,freq)))*w;
            for scale = 1:2
                y_scale = imresize(dpt_filt_frames(:,:,freq), 2 ^ (-(scale -1 )));
                feats_tmp_wpt = [feats_tmp_wpt, GAMIVAL_basic_extractor(y_scale)];
            end
        end

        if log_level == 1, toc; end
        feats_per_frame = [feats_per_frame, feats_tmp_wpt];
        feats_frames(end+1,:) = feats_per_frame;
    end
    fclose(test_file);
end

% Read one frame from YUV file
function YUV = YUVread(f, dim, frnum)

    % This function reads a frame #frnum (0..n-1) from YUV file into an
    % 3D array with Y, U and V components
    
    fseek(f, dim(1)*dim(2)*1.5*frnum, 'bof');
    
    % Read Y-component
    Y = fread(f, dim(1)*dim(2), 'uchar');
    if length(Y) < dim(1)*dim(2)
        YUV = [];
        return;
    end
    Y = cast(reshape(Y, dim(1), dim(2)), 'double');
    
    % Read U-component
    U = fread(f, dim(1)*dim(2)/4, 'uchar');
    if length(U) < dim(1)*dim(2)/4
        YUV = [];
        return;
    end
    U = cast(reshape(U, dim(1)/2, dim(2)/2), 'double');
    U = imresize(U, 2.0);
    
    % Read V-component
    V = fread(f, dim(1)*dim(2)/4, 'uchar');
    if length(V) < dim(1)*dim(2)/4
        YUV = [];
        return;
    end    
    V = cast(reshape(V, dim(1)/2, dim(2)/2), 'double');
    V = imresize(V, 2.0);
    
    % Combine Y, U, and V
    YUV(:,:,1) = Y';
    YUV(:,:,2) = U';
    YUV(:,:,3) = V';
end
