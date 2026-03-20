function analyzir_arirls_batch()
bundle_json = getenv('FNIRS_BUNDLE_JSON');
if isempty(bundle_json)
    error('FNIRS_BUNDLE_JSON environment variable was not set.');
end

analyzir_path = getenv('FNIRS_ANALYZIR_PATH');
if ~isempty(analyzir_path)
    addpath(analyzir_path);
    external_dir = fullfile(analyzir_path, 'external');
    demos_dir = fullfile(analyzir_path, 'demos');
    if exist(external_dir, 'dir') == 7
        addpath(genpath(external_dir));
    end
    if exist(demos_dir, 'dir') == 7
        addpath(genpath(demos_dir));
    end
end

if isempty(which('nirs.modules.GLM')) || isempty(which('nirs.modules.Resample')) || isempty(which('nirs.modules.AR_IRLS'))
    error('AnalyzIR / nirs-toolbox module workflow was not found on the MATLAB path.');
end
if isempty(which('Dictionary'))
    error('AnalyzIR Dictionary class was not found on the MATLAB path.');
end

cfg = jsondecode(fileread(bundle_json));

for iFile = 1:numel(cfg.input_mat_files)
    input_file = string(cfg.input_mat_files{iFile});
    output_file = string(cfg.output_csv_files{iFile});

    S = load(input_file);

    times_s = double(S.times_s(:));
    stim_onsets_s = double(S.stim_onsets_s(:));
    stim_durations_s = double(S.stim_durations_s(:));
    stim_amplitudes = double(S.stim_amplitudes(:));
    Y = double(S.Y);

    channel_names = local_to_cellstr(S.channel_names);
    pair_names = local_to_cellstr(S.pair_names);
    chromophores = local_to_cellstr(S.chromophores);
    target_status = local_to_cellstr(S.target_status);

    nuisance_values = [];
    nuisance_n_reg = [];
    nuisance_names = {};
    if isfield(S, 'nuisance_values')
        nuisance_values = double(S.nuisance_values);
    end
    if isfield(S, 'nuisance_n_reg')
        nuisance_n_reg = double(S.nuisance_n_reg(:));
    end
    if isfield(S, 'nuisance_names')
        nuisance_names = S.nuisance_names;
    end

    subject = local_scalar_string(S.subject);
    file_label = local_scalar_string(S.file_label);
    pipeline_label = local_scalar_string(S.pipeline_label);
    backend = local_scalar_string(S.backend);
    hrf_model = local_scalar_string(S.hrf_model);
    solver = local_scalar_string(S.solver);
    amplitude_value = double(S.amplitude_value(1));
    shift_index = double(S.shift_index(1));
    shift_s = double(S.shift_s(1));
    resample_fs = 4.0;
    if isfield(S, 'matlab_resample_fs')
        resample_fs = double(S.matlab_resample_fs(1));
    end

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
        yi = double(Y(:, ch));
        chrom = chromophores{ch};

        try
            data_obj = local_make_single_channel_data(yi, times_s, chrom, ...
                stim_onsets_s, stim_durations_s, stim_amplitudes, ...
                nuisance_values, nuisance_n_reg, nuisance_names, ch);

            resample_job = nirs.modules.Resample();
            resample_job.Fs = resample_fs;
            data_rs = resample_job.run(data_obj);

            glm_job = nirs.modules.GLM();
            glm_job.type = 'AR-IRLS';
            glm_job.basis = local_basis_dictionary(hrf_model);
            Stats = glm_job.run(data_rs);

            row_idx = local_find_task_row(Stats);
            beta_value = local_pick_stat(Stats.beta, row_idx);
            t_value = local_pick_stat(Stats.tstat, row_idx);
            p_value = local_pick_stat(Stats.p, row_idx);
            se_value = local_compute_se(Stats.covb, row_idx);
            dfe_value = local_pick_dfe(Stats.dfe, row_idx);
        catch ME
            warning('AR-IRLS module workflow failed for %s (%s): %s', channel_names{ch}, pipeline_label, ME.message);
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
        chrom_col(ch) = string(chrom);
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

function data_obj = local_make_single_channel_data(yi, times_s, chrom, stim_onsets_s, stim_durations_s, stim_amplitudes, nuisance_values, nuisance_n_reg, nuisance_names, ch)
probe = nirs.core.Probe(zeros(1,3), zeros(1,3), table(1, 1, {char(chrom)}, 'VariableNames', {'source','detector','type'}));
data_obj = nirs.core.Data(yi(:), times_s(:), probe);

task = nirs.design.StimulusEvents();
task.name = 'task';
task.onset = stim_onsets_s(:);
task.dur = stim_durations_s(:);
task.amp = stim_amplitudes(:);
data_obj.stimulus('task') = task;

if ~isempty(nuisance_values) && ~isempty(nuisance_n_reg)
    n_nuis = nuisance_n_reg(ch);
    for k = 1:n_nuis
        reg_vec = double(squeeze(nuisance_values(:, k, ch)));
        if isempty(reg_vec)
            continue;
        end
        reg_name = local_get_nuisance_name(nuisance_names, k, ch);
        stim_vec = nirs.design.StimulusVector();
        stim_vec.name = reg_name;
        stim_vec.time = times_s(:);
        stim_vec.vector = reg_vec(:);
        try
            stim_vec.regressor_no_interest = true;
        catch
        end
        data_obj.stimulus(reg_name) = stim_vec;
    end
end
end

function basis_dict = local_basis_dictionary(hrf_model)
basis_dict = Dictionary();
switch lower(char(hrf_model))
    case {'glover','canonical'}
        basis_dict('default') = nirs.design.basis.Canonical();
    case 'gamma'
        basis_dict('default') = nirs.design.basis.Gamma();
    otherwise
        error('Unsupported MATLAB HRF model for module workflow: %s', hrf_model);
end
end

function idx = local_find_task_row(Stats)
idx = 1;
try
    if isprop(Stats, 'variables') && ~isempty(Stats.variables)
        vars = Stats.variables;
        var_names = vars.Properties.VariableNames;
        cond_col = '';
        candidates = {'cond','condition','Condition','Cond','variable','Variable'};
        for i = 1:numel(candidates)
            if any(strcmp(var_names, candidates{i}))
                cond_col = candidates{i};
                break;
            end
        end
        if ~isempty(cond_col)
            cond_values = string(vars.(cond_col));
            match = find(cond_values == "task", 1, 'first');
            if isempty(match)
                match = find(contains(lower(cond_values), 'task'), 1, 'first');
            end
            if ~isempty(match)
                idx = double(match);
                return;
            end
        end
    end
catch
end
end

function value = local_pick_stat(arr, idx)
arr = double(arr(:));
if isempty(arr)
    value = NaN;
    return;
end
idx = max(1, min(idx, numel(arr)));
value = arr(idx);
end

function value = local_pick_dfe(dfe, idx)
dfe = double(dfe(:));
if isempty(dfe)
    value = NaN;
elseif numel(dfe) == 1
    value = dfe(1);
else
    idx = max(1, min(idx, numel(dfe)));
    value = dfe(idx);
end
end

function se_value = local_compute_se(covb, idx)
se_value = NaN;
try
    if isempty(covb)
        return;
    end
    sz = size(covb);
    if numel(sz) == 2
        se_value = sqrt(double(covb(idx, idx)));
    elseif numel(sz) == 3
        se_value = sqrt(double(covb(idx, idx, 1)));
    else
        se_value = sqrt(double(covb(idx, idx, 1, 1)));
    end
catch
    se_value = NaN;
end
end

function name = local_get_nuisance_name(nuisance_names, k, ch)
name = sprintf('nuis_%02d', k);
try
    if iscell(nuisance_names)
        raw = nuisance_names{k, ch};
    else
        raw = nuisance_names(k, ch);
    end
    raw = char(string(raw));
    if ~isempty(strtrim(raw))
        name = matlab.lang.makeValidName(raw);
    end
catch
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
