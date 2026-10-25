function juniward_channel_shard(repo_root, manifest_path, out_dir, csv_path, ...
                                payload, channel_QF, seed_base)
% Plain J-UNIWARD sent through the same-quality-factor channel, to size an
% error-correcting code for it.
%
% F15 noted that the fair extra baseline for the same-QF channel is J-UNIWARD
% with a code on top: the channel changes only a few dozen coefficients per
% image, so perhaps ordinary embedding plus error correction does as well as a
% purpose-built robust scheme. Before choosing a code we need the raw errors.
%
% Embedding is juniward_embed_shard's, line for line and with the same seeds,
% so the stego image is the one exp13 already produced and already measured.
% Then: recompress at channel_QF (imwrite of the decoded image, as every other
% shard does), extract, and record
%
%   bit_error_rate     raw message bit errors after the channel
%   coef_changed       coefficients the channel altered
%   sym_err_rate       share of 5-bit symbols holding at least one bit error
%   max_block_sym_err  worst count of bad symbols in any block of 31 symbols
%
% A Reed-Solomon code RS(31,k) over 5-bit symbols corrects (31-k)/2 symbol
% errors per block, and whether it decodes depends only on the error pattern,
% not on the data. So an image is delivered by RS(31,k) exactly when
% max_block_sym_err <= (31-k)/2, and one run answers the question for every k.

    if nargin < 7 || isempty(seed_base); seed_base = 0; end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    addpath(genpath(fullfile(repo_root, 'GMAS', 'JPEG_Toolbox')));
    addpath(genpath(fullfile(repo_root, 'GMAS', 'STC3')));
    addpath(genpath(fullfile(repo_root, 'MINICER')));

    covers = read_manifest(manifest_path);
    n = numel(covers);
    fprintf('J-UNIWARD + channel shard: %d covers, payload %g, channel QF %d\n', ...
            n, payload, channel_QF);

    stego_dir = fullfile(out_dir, 'stego');
    chan_dir  = fullfile(out_dir, 'afterchannel');
    for d = {stego_dir, chan_dir}
        if ~exist(d{1}, 'dir'); mkdir(d{1}); end
    end

    fid = fopen(csv_path, 'w');
    fprintf(fid, ['cover,nzAC,msg_bits,bit_error_rate,coef_changed,sym_err_rate,' ...
                  'max_block_sym_err,status,seconds\n']);
    closer = onCleanup(@() fclose(fid));

    t_all = tic;
    for i = 1:n
        t0 = tic;
        cover_path = covers{i};
        [~, base, ~] = fileparts(cover_path);
        stego_path = fullfile(stego_dir, [base '.jpg']);
        chan_path  = fullfile(chan_dir,  [base '.jpg']);

        status = 'ok'; ber = NaN; nzAC = NaN; msg_len = NaN;
        changed = NaN; ser = NaN; worst = NaN;
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

            imwrite(imread(stego_path), chan_path, 'quality', channel_QF);
            R_STRUCT = jpeg_read(chan_path);
            recv = R_STRUCT.coef_arrays{1};
            changed = nnz(recv ~= S_STRUCT.coef_arrays{1});

            dec = stc_ml_extract(int32(reshape(recv, 1, [])), n_bits, H);
            m = min(numel(msg), numel(dec));
            e = double(msg(1:m)) ~= double(reshape(dec(1:m), 1, []));
            ber = sum(e) / m;

            % 5-bit symbols, then blocks of 31 symbols (the last block shortened).
            e5 = [e, false(1, mod(-m, 5))];
            sym = any(reshape(e5, 5, []), 1);
            ser = mean(sym);
            s31 = [sym, false(1, mod(-numel(sym), 31))];
            worst = max(sum(reshape(s31, 31, []), 1));

        catch err
            status = strrep(strrep(err.message, ',', ';'), newline, ' ');
            if numel(status) > 90; status = status(1:90); end
        end

        if isnan(ber)
            fprintf(fid, '%s,%d,%d,,,,,%s,%.2f\n', base, nzAC, msg_len, status, toc(t0));
            fprintf('  [%d/%d] %s  FAILED: %s\n', i, n, base, status);
        else
            fprintf(fid, '%s,%d,%d,%.8g,%d,%.6g,%d,%s,%.2f\n', ...
                    base, nzAC, msg_len, ber, changed, ser, worst, status, toc(t0));
            if mod(i, 25) == 0
                fprintf('  [%d/%d] %s  ber=%.4g  changed=%d  worst=%d  %.1fs\n', ...
                        i, n, base, ber, changed, worst, toc(t0));
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
