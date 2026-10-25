function pmas_embed_shard(repo_root, manifest_path, out_dir, csv_path, ...
                          payload, cover_QF, attack_QF, seed_base, ...
                          T, mode, distortion, faithful)
% One shard of PMAS (UNICORE) embedding + channel + extraction, per-image CSV.
%
% Mirrors testRobustUNICORE.m -- "Robust Steganography for High Quality Images",
% IEEE TCSVT. The clone is never modified.
%
% THE SHIPPED DRIVER FAILS ON EVERY IMAGE AND REPORTS IT AS PERFECT.
% testRobustUNICORE.m line 52 reads:
%
%     imwrite(imread(stego_Path),afterchannel_stego_Path,'quality',attack_QF);
%
% but the loop defines `stego_path` and `afterchannel_stego_path` (lines 28-29,
% lowercase p). MATLAB is case-sensitive, so this throws "Unrecognized function
% or variable 'stego_Path'" on every iteration. The catch at line 60 then sets
% bit_error_rate(i)=0, and line 67 averages those zeros into
% `ave_error_rate: 0` -- flawless robustness, with no extraction ever performed.
%
% The line is also redundant: generateStegoUNICORE already writes the
% after-channel file via postprocessUNICORE (generateStegoUNICORE.m line 22).
% Dropping it makes the pipeline run.
%
% Pass faithful=true to reproduce the shipped behaviour and watch every image
% fail; faithful=false (default) omits the dead line and actually measures.
%
% Failures are recorded as failures, never as BER=0.
%
% PARAMETERS ARE ASSUMPTIONS. The repo README documents their meaning but not
% the values used for the published results:
%   T          threshold controlling the robust region
%   mode(1)    1 = uncompressed cover, 2 = post-compressed
%   mode(2)    distortion calculation method
%   mode(3)    number of post-processing rounds
%   distortion 1 = J-UNIWARD, 0 = UERD
% Sweep before quoting any number.

    if nargin < 8  || isempty(seed_base);  seed_base = 0;      end
    if nargin < 9  || isempty(T);          T = 8;              end
    if nargin < 10 || isempty(mode);       mode = [2 1 1];     end
    if nargin < 11 || isempty(distortion); distortion = 1;     end
    if nargin < 12 || isempty(faithful);   faithful = false;   end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    addpath(genpath(fullfile(repo_root, 'GMAS', 'JPEG_Toolbox')));
    addpath(genpath(fullfile(repo_root, 'GMAS', 'STC3')));
    addpath(genpath(fullfile(repo_root, 'PMAS')));
    % Patched copies FIRST so they shadow the clone. Two fixes, both the
    % same defect: preprocessUNICORE:27 and postprocessUNICORE:9 call the
    % 5-argument JPEGrecompress(input,output,QF,save,color) with only 3
    % arguments, so `color` is undefined and line 4's `if color` throws on
    % every image. Values are unambiguous: color=0 (BOSSbase is grayscale,
    % and the else-branch is the grayscale path), save=1 (callers read the
    % written file back). See baselines/patched/PMAS/README.md.
    patched = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                       'baselines', 'patched', 'PMAS');
    if exist(patched, 'dir'); addpath(patched, '-begin'); end

    covers = read_manifest(manifest_path);
    n = numel(covers);
    fprintf('PMAS shard: %d covers, payload %g, QF %d -> %d, T %g, mode [%s], dist %g, faithful %d\n', ...
            n, payload, cover_QF, attack_QF, T, num2str(mode), distortion, faithful);

    dct0 = zeros(8,8);
    spatail = zeros(8,8,64);
    for i = 1:8
        for j = 1:8
            dct = dct0;
            dct(i,j) = dct(i,j) + 1;
            fun = @(x) dct2(x.data);
            spatail(:,:,j+(i-1)*8) = ...
                blockproc(double(dct .* quantizationTable(cover_QF)), [8 8], fun);
        end
    end

    stego_dir = fullfile(out_dir, 'stego');
    chan_dir  = fullfile(out_dir, 'afterchannel');
    ccov_dir  = fullfile(out_dir, 'afterchannel_cover');
    for d = {stego_dir, chan_dir, ccov_dir}
        if ~exist(d{1}, 'dir'); mkdir(d{1}); end
    end

    fid = fopen(csv_path, 'w');
    fprintf(fid, 'cover,msg_bits,bit_error_rate,status,psnr_db,seconds\n');
    closer = onCleanup(@() fclose(fid));

    t_all = tic;
    for i = 1:n
        t0 = tic;
        cover_path = covers{i};
        [~, base, ~] = fileparts(cover_path);
        stego_path = fullfile(stego_dir, [base '.jpg']);
        afterchannel_stego_path = fullfile(chan_dir, [base '.jpg']);
        afterchannel_cover_path = fullfile(ccov_dir, [base '.jpg']);

        status = 'ok'; ber = NaN; msg_len = NaN; p = NaN;
        try
            [cover, rhoM, rhoP, modification] = preprocessUNICORE( ...
                cover_path, cover_QF, attack_QF, T, mode, ...
                afterchannel_cover_path, spatail, distortion);

            rng(seed_base + i, 'twister');
            [msg, msg_len] = generateRandMsg(cover_path, payload);

            cover1 = int32(reshape(cover, 1, []));
            costs = zeros(3, size(cover1, 2), 'single');
            costs(1,:) = reshape(rhoM, 1, []);
            costs(3,:) = reshape(rhoP, 1, []);

            H = 10;
            [~, stc_msg, stc_n_msg_bits, ~] = ...
                stc_pm1_pls_embed(int32(cover1), costs, uint8(msg), H);
            stego = reshape(stc_msg, size(cover,1), size(cover,2));

            generateStegoUNICORE(cover_path, stego_path, cover, modification, ...
                stego, afterchannel_stego_path, attack_QF, T, mode(3));

            if faithful
                % Verbatim line 52, including the capitalisation that breaks it.
                imwrite(imread(stego_Path), afterchannel_stego_Path, ...
                        'quality', attack_QF); %#ok<NODEF>
            end

            stc_decoded_msg = stcExtractUNICORE(afterchannel_stego_path, stc_n_msg_bits);

            ber = sum(abs(double(msg) - double(stc_decoded_msg))) / msg_len;
            p = local_psnr(imread(cover_path), imread(stego_path));

        catch err
            status = strrep(strrep(err.message, ',', ';'), newline, ' ');
            if numel(status) > 90; status = status(1:90); end
        end

        if isnan(ber)
            fprintf(fid, '%s,%d,,%s,,%.2f\n', base, msg_len, status, toc(t0));
            fprintf('  [%d/%d] %s  FAILED: %s\n', i, n, base, status);
        else
            fprintf(fid, '%s,%d,%.8g,%s,%.4f,%.2f\n', ...
                    base, msg_len, ber, status, p, toc(t0));
            fprintf('  [%d/%d] %s  ber=%.6g  %.1fs\n', i, n, base, ber, toc(t0));
        end
    end

    fprintf('shard done: %d covers in %.1f s (%.2f s/image)\n', ...
            n, toc(t_all), toc(t_all) / max(n, 1));
end


function paths_out = read_manifest(p)
    fid = fopen(p, 'r');
    if fid < 0; error('cannot open manifest %s', p); end
    c = onCleanup(@() fclose(fid));
    paths_out = {};
    while true
        line = fgetl(fid);
        if ~ischar(line); break; end
        line = strtrim(line);
        if ~isempty(line); paths_out{end+1} = line; end %#ok<AGROW>
    end
end


function v = local_psnr(a, b)
    a = double(a); b = double(b);
    mse = mean((a(:) - b(:)).^2);
    if mse == 0; v = Inf; else; v = 10 * log10(255^2 / mse); end
end
