function juniward_embed_shard(repo_root, manifest_path, out_dir, csv_path, ...
                              payload, seed_base)
% Plain J-UNIWARD embedding — the non-robust security ceiling.
%
% This is the reference point the whole comparison needs. J-UNIWARD is the
% standard content-adaptive JPEG scheme: it minimises a wavelet-domain
% distortion with no robustness mechanism at all, so it marks the *best
% achievable security* at a given payload. Every robust method trades some of
% that away to survive a channel; the size of that trade is Paper 1's subject.
%
% Uses the repo's `J_UNIWARDcost` (byte-identical in MINICER/ and ROAST/) plus
% ternary STC at H=10 — the same coding the robust baselines use, so the only
% difference between this and GMAS/ROAST is the cost function and the absence of
% dither modulation.
%
% No channel is applied and no ECC is used: this is the clean-cover reference.
% Its extraction error is 0 by construction, so the CSV records only what is
% needed downstream (payload actually embedded, PSNR).

    if nargin < 6 || isempty(seed_base); seed_base = 0; end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    addpath(genpath(fullfile(repo_root, 'GMAS', 'JPEG_Toolbox')));
    addpath(genpath(fullfile(repo_root, 'GMAS', 'STC3')));
    addpath(genpath(fullfile(repo_root, 'MINICER')));

    covers = read_manifest(manifest_path);
    n = numel(covers);
    fprintf('J-UNIWARD shard: %d covers, payload %g bpnzac\n', n, payload);

    stego_dir = fullfile(out_dir, 'stego');
    if ~exist(stego_dir, 'dir'); mkdir(stego_dir); end

    fid = fopen(csv_path, 'w');
    fprintf(fid, 'cover,nzAC,msg_bits,bit_error_rate,status,psnr_db,seconds\n');
    closer = onCleanup(@() fclose(fid));

    t_all = tic;
    for i = 1:n
        t0 = tic;
        cover_path = covers{i};
        [~, base, ~] = fileparts(cover_path);
        stego_path = fullfile(stego_dir, [base '.jpg']);

        status = 'ok'; ber = NaN; nzAC = NaN; msg_len = NaN; p = NaN;
        try
            C_STRUCT = jpeg_read(cover_path);
            C_COEFFS = C_STRUCT.coef_arrays{1};
            nzAC = nnz(C_COEFFS) - nnz(C_COEFFS(1:8:end, 1:8:end));

            rng(seed_base + i, 'twister');
            msg_len = ceil(payload * nzAC);
            msg = round(rand(1, msg_len));

            [rhoP1, rhoM1] = J_UNIWARDcost(cover_path);

            cover = int32(reshape(C_COEFFS, 1, []));
            costs = zeros(3, size(cover, 2), 'single');
            costs(1,:) = reshape(rhoM1, 1, []);
            costs(3,:) = reshape(rhoP1, 1, []);

            H = 10;
            [~, stc_msg, n_bits, ~] = ...
                stc_pm1_pls_embed(cover, costs, uint8(msg), H);

            S_STRUCT = C_STRUCT;
            S_STRUCT.coef_arrays{1} = double(reshape(stc_msg, size(C_COEFFS)));
            jpeg_write(S_STRUCT, stego_path);

            % Sanity: no channel, so extraction must be exact.
            dec = stc_ml_extract(int32(stc_msg), n_bits, H);
            m = min(numel(msg), numel(dec));
            ber = sum(abs(double(msg(1:m)) - double(dec(1:m)))) / m;

            p = local_psnr(imread(cover_path), imread(stego_path));

        catch err
            status = strrep(strrep(err.message, ',', ';'), newline, ' ');
            if numel(status) > 90; status = status(1:90); end
        end

        if isnan(ber)
            fprintf(fid, '%s,%d,%d,,%s,,%.2f\n', base, nzAC, msg_len, status, toc(t0));
            fprintf('  [%d/%d] %s  FAILED: %s\n', i, n, base, status);
        else
            fprintf(fid, '%s,%d,%d,%.8g,%s,%.4f,%.2f\n', ...
                    base, nzAC, msg_len, ber, status, p, toc(t0));
            if mod(i, 25) == 0
                fprintf('  [%d/%d] %s  ber=%.6g  %.1fs\n', i, n, base, ber, toc(t0));
            end
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
