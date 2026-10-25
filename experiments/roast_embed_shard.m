function roast_embed_shard(repo_root, manifest_path, out_dir, csv_path, ...
                           payload, cover_QF, attack_QF, seed_base, alpha, beta)
% One shard of ROAST-URA embedding + channel + extraction, per-image CSV out.
%
% Mirrors TestRobustnessURA_JUNIWARD.m: PreprocessURA turns a *precover* into an
% overflow-suppressed *cover*, J-UNIWARD asymmetric cost is reweighted by CostUR
% using the vulnerable-block map, then ternary STC embeds and JPEG recompression
% is the channel. The clone is never modified; this only calls into it.
%
% TWO DELIBERATE DEVIATIONS from the published driver, both to avoid overstating
% robustness:
%
%   1. Their catch block sets bit_error_rate(i)=0 when STC embedding or
%      extraction throws, and that zero is then averaged in -- so a hard failure
%      is scored as a *perfect* result. We record failures as failures and keep
%      them out of the BER mean. (MINICER's Test_QMAS has the same pattern.)
%   2. Their `bit_error_rate` preallocation is commented out while still being
%      indexed inside parfor. We preallocate.
%
% Message RNG is seeded per image so runs reproduce; the published driver calls
% bare rand().

    if nargin < 8 || isempty(seed_base); seed_base = 0; end
    % CostUR's published driver passes alpha=0, beta=1, which makes the whole
    % function a no-op: line 32 adds 0*mean and line 35 multiplies by 1, so the
    % costs pass through untouched and ROAST's cost-adjustment contributes
    % nothing. CostUR's own comments suggest 0.2 and 0.7. Both are exposed here
    % so the difference can be measured rather than argued about.
    if nargin < 9 || isempty(alpha); alpha = 0; end
    if nargin < 10 || isempty(beta); beta = 1; end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    % ROAST first so its J_UNIWARD_Asy_cost / rs_*_yxz shadow GMAS's copies;
    % GMAS is on the path only for the JPEG toolbox, which ROAST lacks.
    addpath(genpath(fullfile(repo_root, 'GMAS', 'JPEG_Toolbox')));
    addpath(genpath(fullfile(repo_root, 'ROAST')));

    precovers = read_manifest(manifest_path);
    n = numel(precovers);
    fprintf('shard: %d precovers, payload %g, QF %d -> %d\n', ...
            n, payload, cover_QF, attack_QF);

    cover_dir = fullfile(out_dir, 'cover');    % PreprocessURA output
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
            C_QUANT = PC.quant_tables{1};
            nzAC = nnz(PC_COEFFS) - nnz(PC_COEFFS(1:8:end, 1:8:end));

            % Overflow suppression: precover -> cover.
            VulnerableBlock = PreprocessURA(precover_Path, cover_Path);

            rng(seed_base + i, 'twister');
            raw_msg_len = ceil(payload * nzAC);
            raw_msg = round(rand(1, raw_msg_len));

            nn = 31; kk = 15; mm = 5;
            pad_n = ceil(raw_msg_len / kk / mm) * kk * mm - raw_msg_len;
            padded = zeros(1, raw_msg_len + pad_n);
            padded(1:raw_msg_len) = raw_msg;
            rs_encoded_msg = rs_encode_yxz(padded, nn, kk);

            [rho1_P, rho1_M] = J_UNIWARD_Asy_cost(cover_Path);
            [rho_p, rho_m] = CostUR(rho1_P, rho1_M, VulnerableBlock, cover_Path, alpha, beta);

            stc_n_msg_bits = stc3Embed(cover_Path, stego_Path, rho_p, rho_m, rs_encoded_msg);

            imwrite(imread(stego_Path), chan_Path, 'quality', attack_QF);

            stc_decoded_msg = stc3Extract(chan_Path, stc_n_msg_bits, C_QUANT);
            rs_decoded_msg  = rs_decode_yxz(double(stc_decoded_msg), nn, kk);
            extracted = rs_decoded_msg(1:raw_msg_len);

            ber = sum(abs(double(raw_msg) - double(extracted))) / raw_msg_len;
            p = local_psnr(imread(precover_Path), imread(stego_Path));

        catch err
            % Recorded as a failure, NOT as ber=0. See header note 1.
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
