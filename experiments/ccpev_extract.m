function ccpev_extract(manifest_path, out_path, QF)
% Extract CC-PEV features (548-dim) for every JPEG in a manifest.
%
% Second, independent feature set for cross-checking the DCTR results. CC-PEV
% is a different family entirely -- 274 DCT-domain features plus their
% Cartesian-calibrated counterparts (decompress, crop 4x4, recompress) -- so
% agreement between it and DCTR is real corroboration rather than two views of
% the same statistic.
%
% Same binary output format as dctr_extract.m: float64 (n x 548) plus a sidecar
% of image stems in order.

    if nargin < 3 || isempty(QF); QF = 65; end

    here = fileparts(mfilename('fullpath'));
    root = fileparts(here);
    addpath(fullfile(root, 'baselines', 'features'));
    addpath(genpath(fullfile(root, 'baselines', 'Robust_Steganography', ...
                             'GMAS', 'JPEG_Toolbox')));

    files = read_manifest(manifest_path);
    n = numel(files);
    fprintf('CCPEV: %d images at QF %d -> %s\n', n, QF, out_path);

    fid = fopen(out_path, 'w');
    closer = onCleanup(@() fclose(fid));
    stems = cell(1, n);

    t0 = tic;
    for i = 1:n
        [~, stems{i}, ~] = fileparts(files{i});
        F = ccpev548(files{i}, QF);
        fwrite(fid, double(F(:)), 'double');
        if mod(i, 50) == 0
            fprintf('  %d/%d  (%.2f s/image)\n', i, n, toc(t0)/i);
        end
    end

    sfid = fopen([out_path '.stems.txt'], 'w');
    fprintf(sfid, '%s\n', stems{:});
    fclose(sfid);
    fprintf('done: %d x %d in %.1f s (%.2f s/image)\n', ...
            n, numel(F), toc(t0), toc(t0)/n);
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
