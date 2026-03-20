function analyzir_arirls_batch()
bundle_json = getenv('FNIRS_BUNDLE_JSON');
if isempty(bundle_json)
    error('FNIRS_BUNDLE_JSON environment variable was not set.');
end

analyzir_path = getenv('FNIRS_ANALYZIR_PATH');
if ~isempty(analyzir_path)
    addpath(genpath(analyzir_path));
end

if exist('nirs', 'dir') ~= 7 && exist('nirs', 'class') ~= 8
    error('AnalyzIR / nirs-toolbox was not found on the MATLAB path.');
end

cfg = jsondecode(fileread(bundle_json));

for iFile = 1:numel(cfg.input_mat_files)
    input_file = string(cfg.input_mat_files{iFile});
    output_file = string(cfg.output_csv_files{iFile});

    S = load(input_file);

    times_s = double(S.times_s(:));
    Y = double(S.Y);
    X = double(S.X);
    n_reg = double(S.n_reg(:));
    task_reg_index = double(S.task_reg_index(:));

    channel_names = local_to_cellstr(S.channel_names);
    pair_names = local_to_cellstr(S.pair_names);
    chromophores = local_to_cellstr(S.chromophores);
    target_status = local_to_cellstr(S.target_status);

    subject = local_scalar_string(S.subject);
    file_label = local_scalar_string(S.file_label);
    pipeline_label = local_scalar_string(S.pipeline_label);
    backend = local_scalar_string(S.backend);
    hrf_model = local_scalar_string(S.hrf_model);
    solver = local_scalar_string(S.solver);
    amplitude_value = double(S.amplitude_value(1));
    shift_index = double(S.shift_index(1));
    shift_s = double(S.shift_s(1));

    Fs = 1 / mean(diff(times_s));
    Pmax = ceil(4 * Fs);

    n_channels = size(Y, 2);

    subject_col = strings(n_channels, 1);
    file_col = strings(n_channels, 1);
    pipe_col = strings(n_channels, 1);
    backend_col = strings(n_channels, 1);
    hrf_col = strings(n_channels, 1);
    solver_col = strings(n_channels, 1);
    chan_col = strings(n_channels, 1);
    pair_col = strings(n_channels, 1);
    chrom_col = strings(n_channels, 1);
    target_col = strings(n_channels, 1);
    task_reg_col = strings(n_channels, 1);
    beta_col = nan(n_channels, 1);
    se_col = nan(n_channels, 1);
    t_col = nan(n_channels, 1);
    p_col = nan(n_channels, 1);
    dfe_col = nan(n_channels, 1);
    amp_col = zeros(n_channels, 1);
    shift_index_col = zeros(n_channels, 1);
    shift_s_col = zeros(n_channels, 1);

    for ch = 1:n_channels
        Xi = squeeze(X(:, 1:n_reg(ch), ch));
        yi = Y(:, ch);

        try
            stats = nirs.math.ar_irls(yi, Xi, Pmax, [], false, false, false);
            beta_value = stats.beta(task_reg_index(ch), 1);
            covb = stats.covb;
            if ndims(covb) == 2
                se_value = sqrt(covb(task_reg_index(ch), task_reg_index(ch)));
            elseif ndims(covb) == 3
                se_value = sqrt(covb(task_reg_index(ch), task_reg_index(ch), 1));
            else
                se_value = sqrt(covb(task_reg_index(ch), task_reg_index(ch), 1, 1));
            end
            dfe_value = stats.dfe;
            if numel(dfe_value) > 1
                dfe_value = dfe_value(1);
            end
            t_value = beta_value / se_value;
            p_value = 2 * tcdf(-abs(t_value), dfe_value);
        catch ME
            warning('AR-IRLS failed for %s (%s): %s', channel_names{ch}, pipeline_label, ME.message);
            beta_value = NaN;
            se_value = NaN;
            t_value = NaN;
            p_value = NaN;
            dfe_value = NaN;
        end

        subject_col(ch) = string(subject);
        file_col(ch) = string(file_label);
        pipe_col(ch) = string(pipeline_label);
        backend_col(ch) = string(backend);
        hrf_col(ch) = string(hrf_model);
        solver_col(ch) = string(solver);
        chan_col(ch) = string(channel_names{ch});
        pair_col(ch) = string(pair_names{ch});
        chrom_col(ch) = string(chromophores{ch});
        target_col(ch) = string(target_status{ch});
        task_reg_col(ch) = "task";
        beta_col(ch) = beta_value;
        se_col(ch) = se_value;
        t_col(ch) = t_value;
        p_col(ch) = p_value;
        dfe_col(ch) = dfe_value;
        amp_col(ch) = amplitude_value;
        shift_index_col(ch) = shift_index;
        shift_s_col(ch) = shift_s;
    end

    T = table(subject_col, file_col, amp_col, pipe_col, backend_col, hrf_col, solver_col, ...
              chan_col, pair_col, chrom_col, target_col, task_reg_col, ...
              beta_col, se_col, t_col, p_col, dfe_col, shift_index_col, shift_s_col, ...
              'VariableNames', {'subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', ...
                                'hrf_model', 'solver', 'channel_name', 'pair_name', 'chromophore', ...
                                'target_status', 'task_regressor', 'beta', 'se', 't_value', 'p_value', ...
                                'dfe', 'shift_index', 'shift_s'});

    writetable(T, output_file);
end
end

function out = local_to_cellstr(x)
if iscell(x)
    out = cell(size(x));
    for i = 1:numel(x)
        out{i} = char(string(x{i}));
    end
elseif isstring(x)
    out = cellstr(x);
elseif ischar(x)
    out = cellstr(x);
else
    out = cellstr(string(x));
end
end

function out = local_scalar_string(x)
if iscell(x)
    out = char(string(x{1}));
elseif isstring(x)
    out = char(x(1));
elseif ischar(x)
    out = x;
else
    out = char(string(x(1)));
end
end
