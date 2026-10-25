function minicer_embed_shard(repo_root, manifest_path, out_dir, csv_path, ...
                             payload, cover_QF, attack_QF, seed_base, step_scale)
% One shard of MINICER (QMAS) embedding + channel + extraction, per-image CSV.
%
% Mirrors Test_QMAS.m: preprocessQIM applies quantization-index-modulation
% preprocessing and returns asymmetric costs plus a wet-element rate, then
% ternary STC embeds, generateStegoQMAS writes the stego JPEG, and JPEG
% recompression is the channel. The clone is never modified.
%
% Unlike GMAS and ROAST, MINICER *computes* the quantization table from the QF
% via quantizationTable() rather than reading it from the file. Verified
% 2026-08-07: quantizationTable(65) and quantizationTable(71) are identical to
% the tables in our covers and in a QF71 recompression, so the two conventions
% agree and MINICER can share the same cover set.
%
% DEVIATIONS from the published driver, both to avoid overstating robustness:
%
%   1. Test_QMAS's catch block sets bit_error_rate(i)=0 on any exception and
%      averages it in, so a hard failure scores as a perfect result. We record
%      failures as failures. (ROAST's driver has the same defect.)
%   2. Test_QMAS assigns `wet_rate` only inside the try block but reads it
%      unconditionally at line 75, so if the first image fails the run dies with
%      "Unrecognized function or variable 'wet_rate'" -- masking the real error.
%      We initialise it.
%
% Message RNG is seeded per image for reproducibility. Note generateRandMsg
% derives the message length from the cover itself, so payload maps to bits the
% same way the published driver intends.

    if nargin < 8 || isempty(seed_base); seed_base = 0; end
    % stego_step is the QIM step size per DCT position. The shipped driver
    % hardcodes ones(8,8) -- step 1, the least robust setting available -- so
    % it is exposed here rather than assumed.
    if nargin < 9 || isempty(step_scale); step_scale = 1; end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    addpath(genpath(fullfile(repo_root, 'GMAS', 'JPEG_Toolbox')));
    addpath(genpath(fullfile(repo_root, 'MINICER')));

    covers = read_manifest(manifest_path);
    n = numel(covers);
    fprintf('MINICER shard: %d covers, payload %g, QF %d -> %d\n', ...
            n, payload, cover_QF, attack_QF);

    % DCT basis in the spatial domain, scaled by the cover quantization table.
    % Lifted verbatim from Test_QMAS lines 7-16; preprocessQIM expects it.
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
    for d = {stego_dir, chan_dir}
        if ~exist(d{1}, 'dir'); mkdir(d{1}); end
    end

    fid = fopen(csv_path, 'w');
    fprintf(fid, 'cover,msg_bits,bit_error_rate,wet_rate,status,psnr_db,seconds\n');
    closer = onCleanup(@() fclose(fid));

    t_all = tic;
    for i = 1:n
        t0 = tic;
        cover_Path = covers{i};
        [~, base, ~] = fileparts(cover_Path);
        stego_Path = fullfile(stego_dir, [base '.jpg']);
        chan_Path  = fullfile(chan_dir,  [base '.jpg']);

        status = 'ok'; ber = NaN; msg_len = NaN; wetratei = NaN; p = NaN;
        try
            stego_step = step_scale * ones(8,8);
            [cover, rhoM, rhoP, wetratei] = ...
                preprocessQIM(cover_Path, stego_step, cover_QF, attack_QF, spatail);

            rng(seed_base + i, 'twister');
            [msg, msg_len] = generateRandMsg(cover_Path, payload);

            cover1 = int32(reshape(cover, 1, []));
            costs = zeros(3, size(cover1, 2), 'single');
            costs(1,:) = reshape(rhoM, 1, []);
            costs(3,:) = reshape(rhoP, 1, []);

            H = 10;
            [~, stc_msg, stc_n_msg_bits, ~] = ...
                stc_pm1_pls_embed(int32(cover1), costs, uint8(msg), H);
            stego = reshape(stc_msg, size(cover,1), size(cover,2));

            generateStegoQMAS(cover_Path, stego_Path, cover, stego, ...
                              stego_step, cover_QF, attack_QF);

            imwrite(imread(stego_Path), chan_Path, 'quality', attack_QF);

            stc_decoded_msg = stcExtractQMAS(chan_Path, stc_n_msg_bits, ...
                                             cover_QF, stego_step, attack_QF);

            ber = sum(abs(double(msg) - double(stc_decoded_msg))) / msg_len;
            p = local_psnr(imread(cover_Path), imread(stego_Path));

        catch err
            status = strrep(strrep(err.message, ',', ';'), newline, ' ');
            if numel(status) > 90; status = status(1:90); end
        end

        if isnan(ber)
            fprintf(fid, '%s,%d,,%g,%s,,%.2f\n', base, msg_len, wetratei, status, toc(t0));
            fprintf('  [%d/%d] %s  FAILED: %s\n', i, n, base, status);
        else
            fprintf(fid, '%s,%d,%.8g,%g,%s,%.4f,%.2f\n', ...
                    base, msg_len, ber, wetratei, status, p, toc(t0));
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
