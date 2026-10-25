function script_embed_shard(repo_root, manifest_path, out_dir, csv_path, ...
                            payload, cover_QF, attack_QF, seed_base, ...
                            R1, R2, R3, distortion, si_fix, sp_released, blind)
% One shard of SCRIPT (GISS) embedding + channel + extraction, per-image CSV.
%
% Mirrors testRobustGISS.m -- "Toward Secure and Robust Steganography for
% Black-Box Generated Images", Zeng et al., IEEE TIFS 2024. The clone is never
% modified.
%
% Unlike PMAS this needs no patched copy: the only hardcoded Linux path
% (`/public/zengkai/expcode/GISS/compressedCover/...`) and the SLURM
% `parcluster`/`parpool` block both live in the *driver*, which this replaces.
% The path is vestigial anyway -- `afterchannel_cover_path` is assigned at
% testRobustGISS.m:54 and never read.
%
% DISTORTION=1 (J-UNIWARD) IS NOT RUNNABLE FROM THIS REPO.
% costGISS.m:17 calls `J_UNIWARD_cost(spa, coeffs_real, QUANT)` -- three
% arguments, taking in-memory spatial data and real-valued coefficients. Every
% J-UNIWARD implementation shipped in the repo takes a single file path
% (`J_UNIWARDcost(coverPath)`, `J_UNIWARD_Asy_cost(coverpath)`), so none can
% substitute. The repo README acknowledges this class of gap: "If other
% essential functions are missing, please look for them in my other
% repositories."
%
% DISTORTION=2 (UERD) IS runnable: costGISS calls `UERDcost(coeffs_real, QUANT)`,
% which matches MINICER's `UERDcost(dct_coef, q_tab)` exactly. We put MINICER on
% the path to supply it.
%
% SCRIPT applies NO error-correcting code (`rs_encoded_msg = msg;` in the
% driver), so its BER is a RAW channel error rate -- comparable with MINICER,
% not with GMAS/ROAST's post-RS(31,15) figures. See BASELINE_STATUS.md.
%
% Parameters, per the repo README (meanings documented, published values not):
%   R1 = T_gamma, R2 = T_lambda, R3 = T_max, distortion 1=JUNIWARD 2=UERD
% Sweep before quoting any number.
%
% Manifest lines are PRECOVER paths (.pgm); the matching cover .jpg is looked up
% in cover_dir by stem.

    if nargin < 8  || isempty(seed_base);  seed_base = 0;  end
    if nargin < 9  || isempty(R1);         R1 = 0.1;       end
    if nargin < 10 || isempty(R2);         R2 = 0.1;       end
    if nargin < 11 || isempty(R3);         R3 = 2;         end
    if nargin < 12 || isempty(distortion); distortion = 2; end
    % si_fix (optional, exp40): 0 = the released cost code, as in every main
    % result; 1-3 = the patched copy's variants (baselines/patched/SCRIPT).
    if nargin < 13 || isempty(si_fix);     si_fix = 0;     end
    % sp_released (optional, exp48): how the spatial pattern of a unit coefficient
    % change is built for coverSelect's pixel-headroom test.
    %   0 = dct2, as this shard was first written (copied from the MINICER / PMAS
    %       drivers of the repository) and as every SCRIPT result up to 2026-10-11
    %       was produced. This DEVIATES from SCRIPT's own driver.
    %   1 = idct2, as testRobustGISS.m line 16 has it (the released behaviour).
    if nargin < 14 || isempty(sp_released); sp_released = 0; end
    % blind (optional, exp48): also extract as a receiver without the cover would -
    % recompute the selection from the received image - and record whether the
    % coefficient set is the sender's and the bit error rate of that extraction.
    if nargin < 15 || isempty(blind);       blind = 0;       end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    addpath(genpath(fullfile(repo_root, 'GMAS', 'JPEG_Toolbox')));
    addpath(genpath(fullfile(repo_root, 'GMAS', 'STC3')));
    addpath(genpath(fullfile(repo_root, 'MINICER')));   % supplies UERDcost
    addpath(genpath(fullfile(repo_root, 'SCRIPT')), '-begin');
    if si_fix ~= 0
        patched = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                           'baselines', 'patched', 'SCRIPT');
        addpath(patched, '-begin');
    end

    entries = read_manifest(manifest_path);
    n = numel(entries);
    fprintf('SCRIPT shard: %d covers, payload %g, QF %d -> %d, R=[%g %g %g], dist %g\n', ...
            n, payload, cover_QF, attack_QF, R1, R2, R3, distortion);

    dct0 = zeros(8,8);
    spatail = zeros(8,8,64);
    for i = 1:8
        for j = 1:8
            dct = dct0;
            dct(i,j) = dct(i,j) + 1;
            if sp_released
                fun = @(x) idct2(x.data);
            else
                fun = @(x) dct2(x.data);
            end
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
    if blind
        fprintf(fid, 'cover,msg_bits,usable_coeffs,bit_error_rate,status,psnr_db,seconds,blind_ber,blind_set_differs\n');
    else
        fprintf(fid, 'cover,msg_bits,usable_coeffs,bit_error_rate,status,psnr_db,seconds\n');
    end
    closer = onCleanup(@() fclose(fid));

    t_all = tic;
    for i = 1:n
        t0 = tic;
        parts = strsplit(entries{i}, '|');
        pcover_path = strtrim(parts{1});
        cover_path  = strtrim(parts{2});
        [~, base, ~] = fileparts(cover_path);
        stego_path = fullfile(stego_dir, [base '.jpg']);
        chan_path  = fullfile(chan_dir,  [base '.jpg']);

        status = 'ok'; ber = NaN; msg_len = NaN; usable = NaN; p = NaN;
        bber = NaN; bdiff = NaN;
        try
            is_cover = coverSelect(cover_path, R1, R2, spatail);
            usable = nnz(is_cover == 1);

            rng(seed_base + i, 'twister');
            [msg, msg_len] = generateRandMsg(cover_path, payload);

            % The driver's own guard: too long a message cannot be embedded.
            if msg_len >= 1.5 * usable
                error('msg overlength: %d bits vs %d usable coefficients', ...
                      msg_len, usable);
            end

            if si_fix ~= 0
                [rhoP, rhoM] = costGISS(pcover_path, cover_path, R3, is_cover, ...
                                        distortion, si_fix);
            else
                [rhoP, rhoM] = costGISS(pcover_path, cover_path, R3, is_cover, distortion);
            end
            stc_n_msg_bits = generateStegoGISS(rhoM, rhoP, msg, cover_path, ...
                                               stego_path, is_cover, R3);

            imwrite(imread(stego_path), chan_path, 'quality', attack_QF);

            % Extract from the SAME coefficient subset the embedder used.
            %
            % generateStegoGISS.m:9-13 selects `is_cover==1` when R3==0 and the
            % superset `is_cover>0` otherwise. testRobustGISS.m's extraction
            % (line ~92) always uses `is_cover==1`, so whenever R3~=0 the driver
            % extracts from a different index set than it embedded into -- a
            % guaranteed failure, and one the catch-scores-zero pattern then
            % reports as perfect. We mirror the embedder, which is the correct
            % behaviour.
            S_STRUCT = jpeg_read(chan_path);
            stego_coeffs = S_STRUCT.coef_arrays{1};
            if R3 == 0
                cover_index = (find(is_cover == 1))';
            else
                cover_index = (find(is_cover > 0))';
            end
            H = 10;
            stc_decoded_msg = stc_ml_extract(int32(stego_coeffs(cover_index)), ...
                                             stc_n_msg_bits, H);

            m = min(numel(msg), numel(stc_decoded_msg));
            ber = sum(abs(double(msg(1:m)) - double(stc_decoded_msg(1:m)))) / m;
            p = local_psnr(imread(cover_path), imread(stego_path));

            if blind
                % The receiver has only the received image.
                is_rx = coverSelect(chan_path, R1, R2, spatail);
                if R3 == 0
                    rx_index = (find(is_rx == 1))';
                else
                    rx_index = (find(is_rx > 0))';
                end
                bdiff = numel(setxor(rx_index, cover_index));
                bber = 1;
                try
                    d2 = stc_ml_extract(int32(stego_coeffs(rx_index)), stc_n_msg_bits, H);
                    m2 = min(numel(msg), numel(d2));
                    bber = sum(abs(double(msg(1:m2)) - double(d2(1:m2)))) / m2;
                catch
                end
            end

        catch err
            status = strrep(strrep(err.message, ',', ';'), newline, ' ');
            if numel(status) > 90; status = status(1:90); end
        end

        if isnan(ber)
            if blind
                fprintf(fid, '%s,%d,%d,,%s,,%.2f,,\n', base, msg_len, usable, status, toc(t0));
            else
                fprintf(fid, '%s,%d,%d,,%s,,%.2f\n', base, msg_len, usable, status, toc(t0));
            end
            fprintf('  [%d/%d] %s  FAILED: %s\n', i, n, base, status);
        else
            if blind
                fprintf(fid, '%s,%d,%d,%.8g,%s,%.4f,%.2f,%.8g,%d\n', ...
                        base, msg_len, usable, ber, status, p, toc(t0), bber, bdiff);
            else
                fprintf(fid, '%s,%d,%d,%.8g,%s,%.4f,%.2f\n', ...
                        base, msg_len, usable, ber, status, p, toc(t0));
            end
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
