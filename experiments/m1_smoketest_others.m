function m1_smoketest_others(repo_root, out_dir)
% Smoke-test the four non-self-contained baselines WITHOUT editing the clone.
%
% GMAS runs out of the box (see m1_verify_baselines.m). ROAST, MINICER, PMAS and
% SCRIPT were written for an HPC cluster: they assume jpeg_read/jpeg_write and
% STC3 are already on the path, that cover directories exist, and (PMAS) that
% SLURM_CPUS_PER_TASK is set. All of that is fixable from the caller, which is
% what this does — the clone stays pristine.
%
% The exception is SCRIPT, which hardcodes an absolute Linux path; it needs a
% patched copy under baselines/patched/ and is only probed here, not run.

    if nargin < 2 || isempty(out_dir); out_dir = pwd; end
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    log_path = fullfile(out_dir, 'm1_smoketest_log.txt');
    if exist(log_path, 'file'); delete(log_path); end
    diary(log_path); diary on;
    cleanup = onCleanup(@() diary('off'));

    gmas = fullfile(repo_root, 'GMAS');

    % The shared toolchain every other baseline assumes is already present.
    addpath(genpath(fullfile(gmas, 'JPEG_Toolbox')));
    addpath(genpath(fullfile(gmas, 'STC3')));

    % GMAS ships the only bundled covers (10 @ QF65). Reuse them.
    cover_dir = fullfile(gmas, 'cover_dir_QF65');
    cover_QF = 65;
    attack_QF = 71;
    payload = 0.1;

    fprintf('=== M1 smoke-test: non-self-contained baselines ===\n');
    fprintf('covers   : %s (QF %d)\n', cover_dir, cover_QF);
    fprintf('payload  : %g,  attack QF %d\n\n', payload, attack_QF);

    %% MINICER — cover_num is hardcoded to 1, so this is the cheapest probe.
    fprintf('--- MINICER (Test_QMAS) ---\n');
    run_one(fullfile(repo_root, 'MINICER'), out_dir, @() ...
        Test_QMAS(cover_dir, fullfile(out_dir, 'minicer_stego'), ...
                  payload, cover_QF, attack_QF));

    %% PMAS — needs a real parpool size; parpool(NaN) is the off-cluster failure.
    fprintf('\n--- PMAS (testRobustUNICORE) ---\n');
    prev = getenv('SLURM_CPUS_PER_TASK');
    setenv('SLURM_CPUS_PER_TASK', num2str(feature('numcores')));
    restore_env = onCleanup(@() setenv('SLURM_CPUS_PER_TASK', prev));
    fprintf('  set SLURM_CPUS_PER_TASK=%s\n', getenv('SLURM_CPUS_PER_TASK'));
    fprintf('  NOTE: cover_num is hardcoded to 10000; with 10 covers this is\n');
    fprintf('        expected to fail on a missing file. We are checking that it\n');
    fprintf('        gets that far, i.e. that the toolchain resolves.\n');
    run_one(fullfile(repo_root, 'PMAS'), out_dir, @() ...
        testRobustUNICORE(cover_dir, fullfile(out_dir, 'pmas_stego'), ...
                          payload, cover_QF, attack_QF, 8, 1, 1));

    %% SCRIPT — probe only; the hardcoded path makes a real run impossible here.
    fprintf('\n--- SCRIPT (testRobustGISS) ---\n');
    scr = fullfile(repo_root, 'SCRIPT', 'testRobustGISS.m');
    txt = fileread(scr);
    if contains(txt, '/public/zengkai/')
        fprintf('  CONFIRMED hardcoded absolute path in %s\n', scr);
        fprintf('  -> needs a patched copy under baselines/patched/SCRIPT/\n');
    else
        fprintf('  no hardcoded path found; re-check\n');
    end

    fprintf('\n--- ROAST (TestRobustnessURA_JUNIWARD) ---\n');
    fprintf('  needs a PRECOVER dir as well as a cover dir, and cover_num is\n');
    fprintf('  hardcoded to 10000. Deferred until BOSSbase is in data/ and a\n');
    fprintf('  precover set has been generated.\n');

    fprintf('\ndone: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
end


function run_one(dir_to_add, out_dir, fn)
% Run one baseline with its folder on the path, capturing failure detail.
    addpath(dir_to_add);
    old = cd(dir_to_add);
    restore = onCleanup(@() cd(old));
    t0 = tic;
    try
        fn();
        fprintf('  RAN in %.1f s\n', toc(t0));
    catch err
        fprintf('  FAILED after %.1f s: %s\n', toc(t0), err.message);
        if ~isempty(err.stack)
            fprintf('    at %s line %d\n', err.stack(1).name, err.stack(1).line);
        end
    end
end
