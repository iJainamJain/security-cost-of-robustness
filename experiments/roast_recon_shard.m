function roast_recon_shard(repo_root, manifest_path, out_dir, csv_path, ...
                           payload, cover_QF, attack_QF, seed_base, alpha, beta)
% RECONSTRUCTED ROAST: the intended pipeline, not the one the repo ships.
%
% `roast_embed_shard.m` reproduces TestRobustnessURA_JUNIWARD.m faithfully and
% gets 0/30 errorless at BER ~0.11, versus GMAS's 76% errorless at 0.0062. The
% cause is that ROAST/ contains no dither-modulation step at all (grep for
% "gmas|dither" returns nothing), so its driver embeds plain J-UNIWARD costs
% into an overflow-suppressed cover with no robustness mechanism in the loop.
%
% The literature states ROAST "operates within the framework of the GMAS
% algorithm". This script composes what that implies:
%
%     PreprocessURA   (ROAST)  precover -> overflow-suppressed cover
%       -> J_UNIWARD_Asy_cost           asymmetric +-1 costs on the cover
%       -> CostUR       (ROAST)  reweight costs by the vulnerable-block map
%       -> gmas()       (GMAS)   generalized dither modulation over those costs
%       -> stc3_embed   (GMAS)   DM-aware ternary STC
%
% GMAS's and ROAST's copies of J_UNIWARD_Asy_cost.m and rs_encode_yxz.m are
% byte-identical, so path order between them does not matter. The two STC
% wrappers differ in name (stc3_embed vs stc3Embed) so there is no clash; we
% deliberately use GMAS's, because only it carries the DM state.
%
% Failures are recorded as failures, never as BER=0.

    if nargin < 8 || isempty(seed_base); seed_base = 0; end
    if nargin < 9 || isempty(alpha); alpha = 0.2; end   % CostUR's own suggestion
    if nargin < 10 || isempty(beta); beta = 0.7; end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    addpath(genpath(fullfile(repo_root, 'GMAS')));
    addpath(genpath(fullfile(repo_root, 'ROAST')));
    % The published stc3_embed sizes its cost array for 512x512 images only.
    % The patched copy sizes it from the costs passed in; at 512x512 it is
    % identical. See baselines/patched/GMAS/README.md.
    patched = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                       'baselines', 'patched', 'GMAS');
    if exist(patched, 'dir'); addpath(patched, '-begin'); end

    precovers = read_manifest(manifest_path);
    n = numel(precovers);
    fprintf('RECONSTRUCTED ROAST: %d precovers, payload %g, QF %d -> %d, alpha %g beta %g\n', ...
            n, payload, cover_QF, attack_QF, alpha, beta);

    cover_dir = fullfile(out_dir, 'cover');
    stego_dir = fullfile(out_dir, 'stego');
    chan_dir  = fullfile(out_dir, 'afterchannel');
    for d = {cover_dir, stego_dir, chan_dir}
        if ~exist(d{1}, 'dir'); mkdir(d{1}); end
    end

    fid = fopen(csv_path, 'w');
    fprintf(fid, 'cover,nzAC,raw_msg_bits,bit_error_rate,status,psnr_db,seconds\n');
    closer = onCleanup(@() fclose(fid));

    t_all = tic;
    for i = 1:n
        t0 = tic;
        precover_Path = precovers{i};
        [~, base, ~] = fileparts(precover_Path);
        cover_Path = fullfile(cover_dir, [base '.jpg']);
        stego_Path = fullfile(stego_dir, [base '.jpg']);
        chan_Path  = fullfile(chan_dir,  [base '.jpg']);

        status = 'ok'; ber = NaN; nzAC = NaN; raw_msg_len = NaN; p = NaN;
        try
            PC = jpeg_read(precover_Path);
            PC_COEFFS = PC.coef_arrays{1};
            nzAC = nnz(PC_COEFFS) - nnz(PC_COEFFS(1:8:end, 1:8:end));

            VulnerableBlock = PreprocessURA(precover_Path, cover_Path);

            % Quant table from the *cover* PreprocessURA actually wrote.
            C_STRUCT = jpeg_read(cover_Path);
            C_QUANT = C_STRUCT.quant_tables{1};

            rng(seed_base + i, 'twister');
            raw_msg_len = ceil(payload * nzAC);
            raw_msg = round(rand(1, raw_msg_len));

            nn = 31; kk = 15; mm = 5;
            pad_n = ceil(raw_msg_len / kk / mm) * kk * mm - raw_msg_len;
            padded = zeros(1, raw_msg_len + pad_n);
            padded(1:raw_msg_len) = raw_msg;
            rs_encoded_msg = rs_encode_yxz(padded, nn, kk);

            [rho1_P, rho1_M] = J_UNIWARD_Asy_cost(cover_Path);

            % ROAST's contribution: reweight by the vulnerable-block map.
            [rho_ur_p, rho_ur_m] = CostUR(rho1_P, rho1_M, VulnerableBlock, ...
                                          cover_Path, alpha, beta);

            % The step the shipped ROAST driver omits: GMAS dither modulation,
            % fed the ROAST-adjusted costs rather than raw J-UNIWARD ones.
            [cover_round, change_p, change_m, rho_p, rho_m] = ...
                gmas(cover_Path, rho_ur_p, rho_ur_m, C_QUANT);

            [~, stc_n_msg_bits] = stc3_embed(rs_encoded_msg, cover_Path, ...
                cover_round, rho_p, rho_m, change_p, change_m, cover_QF, stego_Path);

            imwrite(imread(stego_Path), chan_Path, 'quality', attack_QF);

            stc_decoded_msg = stc3_extract(chan_Path, stc_n_msg_bits, C_QUANT);
            rs_decoded_msg  = rs_decode_yxz(double(stc_decoded_msg), nn, kk);
            extracted = rs_decoded_msg(1:raw_msg_len);

            ber = sum(abs(double(raw_msg) - double(extracted))) / raw_msg_len;
            p = local_psnr(imread(precover_Path), imread(stego_Path));

        catch err
            status = strrep(strrep(err.message, ',', ';'), newline, ' ');
            if numel(status) > 90; status = status(1:90); end
        end

        if isnan(ber)
            fprintf(fid, '%s,%d,%d,,%s,,%.2f\n', base, nzAC, raw_msg_len, status, toc(t0));
            fprintf('  [%d/%d] %s  FAILED: %s\n', i, n, base, status);
        else
            fprintf(fid, '%s,%d,%d,%.8g,%s,%.4f,%.2f\n', ...
                    base, nzAC, raw_msg_len, ber, status, p, toc(t0));
            fprintf('  [%d/%d] %s  ber=%.6g  %.1fs\n', i, n, base, ber, toc(t0));
        end
    end

    fprintf('shard done: %d precovers in %.1f s (%.2f s/image)\n', ...
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
