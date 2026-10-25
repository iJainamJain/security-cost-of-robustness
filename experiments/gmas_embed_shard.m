function gmas_embed_shard(repo_root, manifest_path, out_dir, csv_path, ...
                          payload, cover_QF, attack_QF, seed_base)
% One shard of GMAS embedding + channel + extraction. Writes a per-image CSV.
%
% Mirrors the pipeline in the repo's Test_Robustness_GMAS_JUNIWARD.m exactly
% (J-UNIWARD asymmetric cost -> generalized dither modulation -> ternary STC3,
% RS(31,15) outer code, JPEG recompression as the channel), but parameterised
% and driven by a file manifest so the work can be sharded across processes.
% The clone is never modified; this only calls into it.
%
% Parallel Computing Toolbox is not installed on this machine, so `parfor` gives
% no speedup. Parallelism comes from running several of these as separate OS
% processes -- see exp03_grid_runner.py.
%
% Args:
%   manifest_path  text file, one absolute cover path per line
%   csv_path       output CSV, one row per image
%   seed_base      RNG seed offset; each image uses seed_base + its row index,
%                  so runs are reproducible and shard boundaries don't collide

    if nargin < 8 || isempty(seed_base); seed_base = 0; end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    addpath(genpath(fullfile(repo_root, 'GMAS')));
    % The published stc3_embed sizes its cost array for 512x512 images only.
    % The patched copy sizes it from the costs passed in; at 512x512 it is
    % identical. See baselines/patched/GMAS/README.md.
    patched = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                       'baselines', 'patched', 'GMAS');
    if exist(patched, 'dir'); addpath(patched, '-begin'); end

    covers = read_manifest(manifest_path);
    n = numel(covers);
    fprintf('shard: %d covers, payload %g, QF %d -> %d\n', n, payload, cover_QF, attack_QF);

    fid = fopen(csv_path, 'w');
    fprintf(fid, 'cover,nzAC,raw_msg_bits,bit_error_rate,embed_ok,psnr_db,seconds\n');
    closer = onCleanup(@() fclose(fid));

    stego_dir = fullfile(out_dir, 'stego');
    chan_dir  = fullfile(out_dir, 'afterchannel');
    if ~exist(stego_dir, 'dir'); mkdir(stego_dir); end
    if ~exist(chan_dir, 'dir');  mkdir(chan_dir);  end

    t_all = tic;
    for i = 1:n
        t0 = tic;
        cover_Path = covers{i};
        [~, base, ~] = fileparts(cover_Path);
        stego_Path = fullfile(stego_dir, [base '.jpg']);
        chan_Path  = fullfile(chan_dir,  [base '.jpg']);

        try
            C_STRUCT = jpeg_read(cover_Path);
            C_COEFFS = C_STRUCT.coef_arrays{1};
            C_QUANT  = C_STRUCT.quant_tables{1};
            nzAC = nnz(C_COEFFS) - nnz(C_COEFFS(1:8:end, 1:8:end));

            % Seeded per image so a re-run reproduces bit-for-bit. The repo's
            % own test calls bare rand(), which is not reproducible.
            rng(seed_base + i, 'twister');

            raw_msg_len = ceil(payload * nzAC);
            raw_msg = round(rand(1, raw_msg_len));

            nn = 31; kk = 15; mm = 5;   % RS(31,15); length must be a multiple of kk*mm
            pad_n = ceil(raw_msg_len / kk / mm) * kk * mm - raw_msg_len;
            padded = zeros(1, raw_msg_len + pad_n);
            padded(1:raw_msg_len) = raw_msg;

            rs_encoded_msg = rs_encode_yxz(padded, nn, kk);

            [rho1_P, rho1_M] = J_UNIWARD_Asy_cost(cover_Path);
            [cover_round, change_p, change_m, rho_p, rho_m] = ...
                gmas(cover_Path, rho1_P, rho1_M, C_QUANT);

            [suc, stc_n_msg_bits] = stc3_embed(rs_encoded_msg, cover_Path, ...
                cover_round, rho_p, rho_m, change_p, change_m, cover_QF, stego_Path);

            % Channel: JPEG recompression at the attack quality factor.
            imwrite(imread(stego_Path), chan_Path, 'quality', attack_QF);

            stc_decoded_msg = stc3_extract(chan_Path, stc_n_msg_bits, C_QUANT);
            rs_decoded_msg  = rs_decode_yxz(double(stc_decoded_msg), nn, kk);
            extracted = rs_decoded_msg(1:raw_msg_len);

            ber = sum(abs(double(raw_msg) - double(extracted))) / raw_msg_len;

            % Perceptual only -- secondary, never a security claim.
            p = local_psnr(imread(cover_Path), imread(stego_Path));

            fprintf(fid, '%s,%d,%d,%.8g,%d,%.4f,%.2f\n', ...
                    base, nzAC, raw_msg_len, ber, suc, p, toc(t0));
            fprintf('  [%d/%d] %s  ber=%.6g  %.1fs\n', i, n, base, ber, toc(t0));

        catch err
            fprintf(fid, '%s,,,,0,,%.2f\n', base, toc(t0));
            fprintf('  [%d/%d] %s  FAILED: %s\n', i, n, base, err.message);
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
