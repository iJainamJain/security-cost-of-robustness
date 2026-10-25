function m1_verify_baselines(repo_root, out_dir)
% M1 gate (PLAN.md): can we reproduce a published robust-steganography number?
%
% Checks the mex binaries load under this MATLAB, then runs the GMAS robustness
% test bundled with github.com/coriverchen/Robust_Steganography:
%   cover QF 65 -> "Facebook" recompression QF 71, payload 0.1 bpnzAC,
%   J-UNIWARD asymmetric cost + generalized dither modulation + ternary STC,
%   RS(31,15) outer code, 10 bundled covers.
%
% Cover QF 65 < channel QF 71 is the "upward recompression" regime, where dither
% modulation is theoretically error-free. The gate is therefore an average bit
% error rate at or very near zero. A materially non-zero rate means something is
% wrong with the toolchain, not with GMAS.
%
% Never edits the cloned repo; only writes stego output alongside it (that tree
% is gitignored) and a log to out_dir.

    if nargin < 1 || isempty(repo_root)
        error('repo_root is required');
    end
    if nargin < 2 || isempty(out_dir)
        out_dir = pwd;
    end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    gmas_dir = fullfile(repo_root, 'GMAS');
    if ~exist(gmas_dir, 'dir')
        error('GMAS folder not found at %s', gmas_dir);
    end

    log_path = fullfile(out_dir, 'm1_gate_log.txt');
    if exist(log_path, 'file'); delete(log_path); end
    diary(log_path); diary on;
    cleanup = onCleanup(@() diary('off'));

    fprintf('=== M1 gate: reproduce a published GMAS number ===\n');
    fprintf('MATLAB      : %s\n', version);
    fprintf('computer    : %s\n', computer);
    fprintf('repo        : %s\n', repo_root);
    fprintf('started     : %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

    old = cd(gmas_dir);
    restore_cd = onCleanup(@() cd(old));
    addpath(genpath(gmas_dir));

    %% 1. mex binaries
    fprintf('--- mex load check ---\n');
    needed = {'jpeg_read', 'jpeg_write', 'stc_pm1_pls_embed', 'stc_ml_extract'};
    for k = 1:numel(needed)
        w = which(needed{k});
        if isempty(w)
            fprintf('  MISS  %s (not on path)\n', needed{k});
        else
            fprintf('  found %s -> %s\n', needed{k}, w);
        end
    end

    % Loading is what actually fails on a MATLAB/mex version mismatch, so call it.
    fprintf('\n--- jpeg_read functional check ---\n');
    probe = fullfile(gmas_dir, 'cover_dir_QF65', '1.jpg');
    try
        S = jpeg_read(probe);
        C = S.coef_arrays{1};
        nzac = nnz(C) - nnz(C(1:8:end, 1:8:end));
        fprintf('  jpeg_read OK: %dx%d coefficients, %d non-zero AC\n', ...
                size(C,1), size(C,2), nzac);
    catch err
        fprintf('  jpeg_read FAILED: %s\n', err.message);
        fprintf('\nGATE: FAIL (mex will not load)\n');
        return;
    end

    %% 2. the published test
    fprintf('\n--- running Test_Robustness_GMAS_JUNIWARD ---\n');
    fprintf('  expected: average bit error rate at or near 0\n\n');
    t0 = tic;
    try
        Test_Robustness_GMAS_JUNIWARD();
        elapsed = toc(t0);
        fprintf('\n  completed in %.1f s\n', elapsed);
        fprintf('\nGATE: test ran to completion. Read ave_error_rate above.\n');
        fprintf('      PASS if it is 0 or very near 0; investigate otherwise.\n');
    catch err
        fprintf('\n  FAILED after %.1f s: %s\n', toc(t0), err.message);
        for k = 1:numel(err.stack)
            fprintf('    at %s line %d\n', err.stack(k).name, err.stack(k).line);
        end
        fprintf('\nGATE: FAIL\n');
    end

    fprintf('\nfinished    : %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
end
