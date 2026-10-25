function dctr_extract(manifest_path, out_path, QF)
% Extract DCTR features (8000-dim) for every JPEG listed in a manifest.
%
% DCTR is Holub & Fridrich's JPEG feature set (DDE Lab, Binghamton), used here
% as shipped -- writing our own would risk a subtly wrong extractor, which is
% very hard to notice and poisons every downstream P_E.
%
% Writes a plain binary float64 matrix (n_images x 8000) plus a sidecar .txt of
% image stems in the same order, so Python can memory-map it without needing a
% MAT-file reader.
%
% Measured 0.54 s/image steady state on this machine.

    if nargin < 3 || isempty(QF); QF = 65; end

    here = fileparts(mfilename('fullpath'));
    root = fileparts(here);
    addpath(fullfile(root, 'baselines', 'features', 'DCTR_matlab_v1.1', 'matlab'));
    addpath(genpath(fullfile(root, 'baselines', 'Robust_Steganography', ...
                             'GMAS', 'JPEG_Toolbox')));

    files = read_manifest(manifest_path);
    n = numel(files);
    fprintf('DCTR: %d images at QF %d -> %s\n', n, QF, out_path);

    fid = fopen(out_path, 'w');
    closer = onCleanup(@() fclose(fid));
    stems = cell(1, n);

    t0 = tic;
    for i = 1:n
        [~, stems{i}, ~] = fileparts(files{i});
        S = jpeg_read(files{i});
        F = DCTR(S, QF);
        fwrite(fid, double(F(:)), 'double');
        if mod(i, 50) == 0
            fprintf('  %d/%d  (%.2f s/image)\n', i, n, toc(t0)/i);
        end
    end

    sidecar = [out_path '.stems.txt'];
    sfid = fopen(sidecar, 'w');
    fprintf(sfid, '%s\n', stems{:});
    fclose(sfid);

    fprintf('done: %d x 8000 in %.1f s (%.2f s/image)\n', n, toc(t0), toc(t0)/n);
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
