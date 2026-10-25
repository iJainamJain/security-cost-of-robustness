function exp47_script_spatail_check(repo, cover_dir, out_csv, cover_QF, n, R)
% Does our SCRIPT harness select the same coefficients as the released driver?
%
% Found by the review council (2026-10-11): script_embed_shard.m builds the
% spatial patterns of a unit coefficient change with dct2, as the MINICER and
% PMAS drivers of the repository do, while SCRIPT's own drivers
% (testRobustGISS.m, line 16) use idct2. The patterns enter coverSelect's
% second test only (pixel headroom, T_lambda): a coefficient is marked 2 or 3
% when a +1 or -1 change would push a pixel of its block out of range.
%
% For the first n covers this computes the selection both ways and counts the
% coefficients whose mark differs. No embedding.
%
%   exp47_script_spatail_check(repo, cover_dir, out_csv, 65, 300, 0)

    addpath(genpath(fullfile(repo, 'GMAS', 'JPEG_Toolbox')));   % as script_embed_shard.m
    addpath(genpath(fullfile(repo, 'GMAS', 'STC3')));
    addpath(genpath(fullfile(repo, 'MINICER')));
    addpath(genpath(fullfile(repo, 'SCRIPT')), '-begin');
    dct0 = zeros(8,8);
    sp_ours = zeros(8,8,64);  sp_rel = zeros(8,8,64);
    qt = quantizationTable(cover_QF);
    for i = 1:8
        for j = 1:8
            d = dct0; d(i,j) = 1;
            sp_ours(:,:,j+(i-1)*8) = dct2(double(d .* qt));
            sp_rel(:,:,j+(i-1)*8)  = idct2(double(d .* qt));
        end
    end
    fid = fopen(out_csv, 'w');
    fprintf(fid, 'cover,coeffs,usable_ours,usable_released,differ,ours1_rel23,ours23_rel1,excluded_first_test\n');
    closer = onCleanup(@() fclose(fid));
    for k = 1:n
        p = fullfile(cover_dir, sprintf('%d.jpg', k));
        a = coverSelect(p, R, R, sp_ours);
        b = coverSelect(p, R, R, sp_rel);
        fprintf(fid, '%d,%d,%d,%d,%d,%d,%d,%d\n', k, numel(a), nnz(a == 1), nnz(b == 1), ...
                nnz(a ~= b), nnz(a == 1 & b > 1), nnz(a > 1 & b == 1), nnz(a == 0));
        if mod(k, 25) == 0; fprintf('%d of %d\n', k, n); end
    end
end
