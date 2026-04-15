function analyzir_glm_native_batch_v7()
% Native AnalyzIR batch helper used by the Python benchmark runner.
% It loads each bundled spec, runs the toolbox-native preprocessing/GLM,
% and writes a flat CSV for downstream aggregation.

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

required_syms = { ...
    'nirs.io.loadSNIRF', ...
    'nirs.modules.Resample', ...
    'nirs.modules.OpticalDensity', ...
    'nirs.modules.BeerLambertLaw', ...
    'nirs.modules.GLM', ...
    'nirs.modules.AR_IRLS', ...
    'nirs.design.extractHRF'};
for i = 1:numel(required_syms)
    if isempty(which(required_syms{i}))
        error('Required AnalyzIR symbol not found on MATLAB path: %s', required_syms{i});
    end
end

cfg = jsondecode(fileread(bundle_json));
for iFile = 1:numel(cfg.input_mat_files)
    input_file = string(cfg.input_mat_files{iFile});
    output_file = string(cfg.output_csv_files{iFile});
    S = load(input_file);

    raw = local_load_snirf(S);
    raw = local_standardize_stimulus(raw, S);
    stim_dict = raw.stimulus;

    hb = local_preprocess_native(raw, S);
    clear raw;

    hb = local_apply_pair_masks(hb, S);
    Stats = local_run_native_glm(hb, S);
    stats_tbl = local_channelstats_to_table(Stats);
    out_tbl = local_build_output_table(stats_tbl, Stats, hb, stim_dict, S);

    writetable(out_tbl, output_file);
    clear S hb Stats stats_tbl stim_dict out_tbl;
end
end

function raw = local_load_snirf(S)
snirf_file = local_scalar_string(S.snirf_file);
if exist(snirf_file, 'file') ~= 2
    error('SNIRF file not found: %s', snirf_file);
end
raw = nirs.io.loadSNIRF(snirf_file, true);
if iscell(raw)
    raw = [raw{:}];
end
if numel(raw) < 1
    error('nirs.io.loadSNIRF returned no data for %s', snirf_file);
end
raw = raw(1);
if ~isa(raw, 'nirs.core.Data')
    error('Expected nirs.core.Data from loadSNIRF, got %s', class(raw));
end
end

function raw = local_standardize_stimulus(raw, S)
shift_s = double(S.shift_s(1));
[onset, dur, amp] = local_collect_existing_stim(raw);

if isempty(onset) && isfield(S, 'stim_onset_s') && ~isempty(S.stim_onset_s)
    onset = double(S.stim_onset_s(:));
    if isfield(S, 'stim_dur_s') && ~isempty(S.stim_dur_s)
        dur = double(S.stim_dur_s(:));
    else
        dur = zeros(size(onset));
    end
    if isfield(S, 'stim_amp') && ~isempty(S.stim_amp)
        amp = double(S.stim_amp(:));
    else
        amp = ones(size(onset));
    end
end

if isempty(onset)
    error('No stimulus events in SNIRF and no fallback schedule provided in MAT bundle.');
end

if shift_s ~= 0
    total_dur = raw.time(end) - raw.time(1);
    if total_dur > 0
        onset = mod((onset - raw.time(1)) + shift_s, total_dur) + raw.time(1);
    else
        onset = onset + shift_s;
    end
end

[onset, order] = sort(onset);
dur = dur(order);
amp = amp(order);

stim = nirs.design.StimulusEvents();
stim.name = 'task';
stim.onset = onset;
stim.dur = dur;
stim.amp = amp;

raw.stimulus = Dictionary();
raw.stimulus('task') = stim;
if isempty(raw.demographics)
    raw.demographics = Dictionary();
end
raw.demographics('subject') = local_scalar_string(S.subject);
raw.demographics('file_label') = local_scalar_string(S.file_label);
raw.demographics('pipeline_label') = local_scalar_string(S.pipeline_label);
end

function [onset, dur, amp] = local_collect_existing_stim(raw)
keys = raw.stimulus.keys;
onset = [];
dur = [];
amp = [];
for i = 1:numel(keys)
    st = raw.stimulus(keys{i});
    if isempty(st)
        continue;
    end
    if isprop(st, 'onset') && ~isempty(st.onset)
        onset = [onset; double(st.onset(:))]; %#ok<AGROW>
        if isprop(st, 'dur') && ~isempty(st.dur)
            dur = [dur; double(st.dur(:))]; %#ok<AGROW>
        else
            dur = [dur; zeros(numel(st.onset), 1)]; %#ok<AGROW>
        end
        if isprop(st, 'amp') && ~isempty(st.amp)
            amp = [amp; double(st.amp(:))]; %#ok<AGROW>
        else
            amp = [amp; ones(numel(st.onset), 1)]; %#ok<AGROW>
        end
    end
end
end

function hb = local_preprocess_native(raw, S)
resample_fs = double(S.analyzir_resample_fs_hz(1));
use_tddr = false;
if isfield(S, 'analyzir_use_tddr')
    use_tddr = logical(double(S.analyzir_use_tddr(1)) ~= 0);
end
ppf_value = 6.0;
if isfield(S, 'ppf_value') && ~isempty(S.ppf_value)
    ppf_value = double(S.ppf_value(1));
end

job = nirs.modules.Resample();
if isfinite(resample_fs) && resample_fs > 0
    job.Fs = resample_fs;
end
job = nirs.modules.OpticalDensity(job);
if use_tddr
    job = nirs.modules.TDDR(job);
end
job = nirs.modules.BeerLambertLaw(job);
job.PPF = ppf_value;

hb = job.run(raw);
if numel(hb) < 1
    error('Native AnalyzIR preprocessing returned no data.');
end
hb = hb(1);
end

function hb = local_apply_pair_masks(hb, S)
pipeline_label = lower(local_scalar_string(S.pipeline_label));
bad_pair_names = {};
short_pair_names = {};
if isfield(S, 'bad_pair_names')
    bad_pair_names = local_to_cellstr(S.bad_pair_names);
end
if isfield(S, 'short_pair_names')
    short_pair_names = local_to_cellstr(S.short_pair_names);
end

link = hb.probe.link;
pair_names = cell(height(link), 1);
for i = 1:height(link)
    pair_names{i} = sprintf('S%d_D%d', double(link.source(i)), double(link.detector(i)));
end

short_mask = ismember(pair_names, short_pair_names);
if ismember('ShortSeperation', link.Properties.VariableNames)
    short_mask = logical(link.ShortSeperation) | short_mask;
end
if ~ismember('ShortSeperation', link.Properties.VariableNames)
    hb.probe.link.ShortSeperation = short_mask;
else
    hb.probe.link.ShortSeperation = logical(hb.probe.link.ShortSeperation) | short_mask;
end

bad_mask = ismember(pair_names, bad_pair_names);
keep_mask = ~bad_mask;
if contains(pipeline_label, 'noss')
    keep_mask = keep_mask & ~short_mask;
end

if ~any(keep_mask)
    error('All channels were removed by native pair masking for pipeline %s.', pipeline_label);
end

hb.data = hb.data(:, keep_mask);
hb.probe.link = hb.probe.link(keep_mask, :);
end

function Stats = local_run_native_glm(hb, S)
pipeline_label = lower(local_scalar_string(S.pipeline_label));
hrf_model = lower(local_scalar_string(S.hrf_model));
solver = lower(strtrim(local_scalar_string(S.solver)));

if contains(pipeline_label, 'localssfilter')
    if isempty(which('advanced.nirs.modules.ShortDistanceFilter'))
        error('advanced.nirs.modules.ShortDistanceFilter not found on MATLAB path.');
    end
    job = advanced.nirs.modules.ShortDistanceFilter();
    job = nirs.modules.GLM(job);
else
    job = nirs.modules.GLM();
end
switch solver
    case 'ols'
        job.type = 'OLS';
    case {'arirls', 'ar-irls'}
        job.type = 'AR-IRLS';
    otherwise
        error('Unsupported native AnalyzIR solver: %s', solver);
end
if contains(pipeline_label, 'localssreg')
    job.AddShortSepRegressors = true;
end
b = Dictionary();
b('default') = local_make_basis(hrf_model);
job.basis = b;
Stats = job.run(hb);
end

function basis = local_make_basis(hrf_model)
switch lower(strtrim(hrf_model))
    case 'canonical'
        basis = nirs.design.basis.Canonical();
    case 'canonical_derivs'
        basis = nirs.design.basis.Canonical();
        if isprop(basis, 'incDeriv')
            basis.incDeriv = true;
        end
        if isprop(basis, 'keepDerivs')
            basis.keepDerivs = true;
        end
    case 'gamma'
        basis = nirs.design.basis.Gamma();
    otherwise
        error('Unsupported native AnalyzIR HRF model: %s', hrf_model);
end
end

function tbl = local_channelstats_to_table(Stats)
if numel(Stats) ~= 1
    error('Expected a single ChannelStats object, got %d.', numel(Stats));
end
try
    tbl = Stats.table();
    if istable(tbl)
        return;
    end
catch
end
try
    tbl = Stats.table;
    if istable(tbl)
        return;
    end
catch
end
error('Could not convert ChannelStats to table.');
end

function out_tbl = local_build_output_table(stats_tbl, Stats, hb, stim_dict, S)
beta_tbl = local_export_stats_table(stats_tbl, hb, S);
curve_tbl = local_export_native_curve_table(Stats, hb, stim_dict, S);
if isempty(beta_tbl) && isempty(curve_tbl)
    out_tbl = local_empty_output_table();
elseif isempty(beta_tbl)
    out_tbl = curve_tbl;
elseif isempty(curve_tbl)
    out_tbl = beta_tbl;
else
    out_tbl = [beta_tbl; curve_tbl];
end
end

function out_tbl = local_export_stats_table(stats_tbl, hb, S)
source_col = local_find_column(stats_tbl, {'source', 'src'});
detector_col = local_find_column(stats_tbl, {'detector', 'det'});
type_col = local_find_column(stats_tbl, {'type', 'hb', 'datatype'});
cond_col = local_find_column(stats_tbl, {'cond', 'condition', 'variable', 'regressor'});
beta_col = local_find_column(stats_tbl, {'beta', 'theta', 'coef', 'estimate'});
t_col = local_find_column(stats_tbl, {'tstat', 't', 't_value'});
p_col = local_find_column(stats_tbl, {'p', 'pval', 'p_value'});
se_col = local_find_column(stats_tbl, {'se', 'stderr', 'std_err'});
dfe_col = local_find_column(stats_tbl, {'dfe', 'df'});
if isempty(source_col) || isempty(detector_col) || isempty(type_col) || isempty(cond_col) || isempty(beta_col)
    error('Could not identify required ChannelStats columns. Columns were: %s', strjoin(stats_tbl.Properties.VariableNames, ', '));
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
meta_map = local_build_link_meta_map(hb.probe.link, S);

rows = cell(0, 26);
for i = 1:height(stats_tbl)
    src = double(stats_tbl.(source_col)(i));
    det = double(stats_tbl.(detector_col)(i));
    typ = lower(strtrim(char(string(stats_tbl.(type_col)(i)))));
    key = local_make_key(src, det, typ);
    if ~isKey(meta_map, key)
        continue;
    end
    meta = meta_map(key);
    if strcmp(meta.target_status, 'short_separation')
        continue;
    end
    cond = strtrim(char(string(stats_tbl.(cond_col)(i))));
    [task_order, is_primary] = local_condition_order(cond);
    beta_val = double(stats_tbl.(beta_col)(i)) * 1e-6;
    se_val = NaN; t_val = NaN; p_val = NaN; dfe_val = NaN;
    if ~isempty(se_col), se_val = double(stats_tbl.(se_col)(i)) * 1e-6; end
    if ~isempty(t_col), t_val = double(stats_tbl.(t_col)(i)); end
    if ~isempty(p_col), p_val = double(stats_tbl.(p_col)(i)); end
    if ~isempty(dfe_col), dfe_val = double(stats_tbl.(dfe_col)(i)); end
    rows(end+1,:) = {subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, ...
        meta.channel_name, meta.pair_name, meta.chromophore, meta.target_status, ...
        'beta', '', NaN, NaN, NaN, cond, task_order, is_primary, ...
        beta_val, se_val, t_val, p_val, dfe_val, shift_index, shift_s}; %#ok<AGROW>
end
out_tbl = local_rows_to_output_table(rows);
end

function out_tbl = local_export_native_curve_table(Stats, hb, stim_dict, S)
subject = local_scalar_string(S.subject);
file_label = local_scalar_string(S.file_label);
pipeline_label = local_scalar_string(S.pipeline_label);
backend = local_scalar_string(S.backend);
hrf_model = local_scalar_string(S.hrf_model);
solver = local_scalar_string(S.solver);
amplitude_value = double(S.amplitude_value(1));
shift_index = double(S.shift_index(1));
shift_s = double(S.shift_s(1));
meta_map = local_build_link_meta_map(hb.probe.link, S);

Fs = NaN;
if isfield(S, 'hrf_export_fs_hz')
    Fs = double(S.hrf_export_fs_hz(1));
elseif isfield(S, 'analyzir_resample_fs_hz')
    Fs = double(S.analyzir_resample_fs_hz(1));
end
if ~isfinite(Fs) || Fs <= 0
    Fs = 1.0;
end

b = Dictionary();
b('default') = local_make_basis(hrf_model);
duration_arg = local_make_extracthrf_duration_arg(stim_dict, Fs);
HRF = nirs.design.extractHRF(Stats, b, duration_arg, Fs, 'hrf');
if numel(HRF) < 1
    out_tbl = local_empty_output_table();
    return;
end
HRF = HRF(1);

curve_time = double(HRF.time(:));
curve_data = HRF.data;
curve_link = HRF.probe.link;
rows = cell(0, 26);
for j = 1:size(curve_data, 2)
    src = double(curve_link.source(j));
    det = double(curve_link.detector(j));
    typ = lower(strtrim(char(string(curve_link.type(j)))));
    typ = regexprep(typ, '_task$', '');
    key = local_make_key(src, det, typ);
    if ~isKey(meta_map, key)
        continue;
    end
    meta = meta_map(key);
    if strcmp(meta.target_status, 'short_separation')
        continue;
    end
    signal = real(double(curve_data(:, j))) * 1e-6;
    signal_se = imag(double(curve_data(:, j))) * 1e-6;
    for k = 1:numel(curve_time)
        rows(end+1,:) = {subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, ...
            meta.channel_name, meta.pair_name, meta.chromophore, meta.target_status, ...
            'hrf_curve', 'analyzir_extractHRF_native', curve_time(k), signal(k), signal_se(k), ...
            'task', 1, true, NaN, NaN, NaN, NaN, NaN, shift_index, shift_s}; %#ok<AGROW>
    end
end
out_tbl = local_rows_to_output_table(rows);
end

function duration_arg = local_make_extracthrf_duration_arg(stim_dict, Fs)
if nargin < 2 || ~isfinite(Fs) || Fs <= 0
    Fs = 1.0;
end
if isempty(stim_dict)
    duration_arg = local_make_scalar_duration_dictionary({'task'}, 1.0 / Fs);
    return;
end
keys = stim_dict.keys;
if isempty(keys)
    duration_arg = local_make_scalar_duration_dictionary({'task'}, 1.0 / Fs);
    return;
end

duration_arg = Dictionary();
for i = 1:numel(keys)
    key = keys{i};
    s = stim_dict(key);
    dur_scalar = NaN;
    if isprop(s, 'dur') && ~isempty(s.dur)
        dur_vals = double(s.dur(:));
        dur_vals = dur_vals(isfinite(dur_vals) & dur_vals > 0);
        if ~isempty(dur_vals)
            dur_scalar = median(dur_vals);
        end
    end
    if ~isfinite(dur_scalar) || dur_scalar <= 0
        dur_scalar = 1.0 / Fs;
    end
    stim = nirs.design.StimulusEvents();
    stim.name = char(string(key));
    stim.onset = 0;
    stim.dur = dur_scalar;
    stim.amp = 1;
    duration_arg(key) = stim;
end
end

function duration_arg = local_make_scalar_duration_dictionary(keys, dur_scalar)
duration_arg = Dictionary();
for i = 1:numel(keys)
    stim = nirs.design.StimulusEvents();
    stim.name = char(string(keys{i}));
    stim.onset = 0;
    stim.dur = dur_scalar;
    stim.amp = 1;
    duration_arg(keys{i}) = stim;
end
end

function meta_map = local_build_link_meta_map(link, S)
target_pair_names = local_to_cellstr(S.target_pair_names);
meta_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
for i = 1:height(link)
    src = double(link.source(i));
    det = double(link.detector(i));
    typ = lower(strtrim(char(string(link.type(i)))));
    pair_name = sprintf('S%d_D%d', src, det);
    is_short = false;
    if ismember('ShortSeperation', link.Properties.VariableNames)
        is_short = logical(link.ShortSeperation(i));
    end
    meta = struct();
    meta.channel_name = sprintf('%s %s', pair_name, typ);
    meta.pair_name = pair_name;
    meta.chromophore = typ;
    if is_short
        meta.target_status = 'short_separation';
    elseif ismember(pair_name, target_pair_names)
        meta.target_status = 'true_target';
    else
        meta.target_status = 'true_non_target';
    end
    meta_map(local_make_key(src, det, typ)) = meta;
end
end

function [order, is_primary] = local_condition_order(cond)
cond = char(string(cond));
if strcmp(cond, 'task')
    order = 1;
    is_primary = true;
    return;
end
m = regexp(cond, ':0?(\d+)$', 'tokens', 'once');
if isempty(m)
    order = 2;
else
    order = 1 + str2double(m{1});
end
is_primary = false;
end

function key = local_make_key(src, det, typ)
key = sprintf('%d|%d|%s', src, det, lower(strtrim(char(string(typ)))));
end

function tbl = local_empty_output_table()
variable_names = local_output_variable_names();
tbl = cell2table(cell(0, numel(variable_names)), 'VariableNames', variable_names);
end

function tbl = local_rows_to_output_table(rows)
variable_names = local_output_variable_names();
if isempty(rows)
    tbl = local_empty_output_table();
else
    tbl = cell2table(rows, 'VariableNames', variable_names);
end
end

function variable_names = local_output_variable_names()
variable_names = {'subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', ...
    'hrf_model', 'solver', 'channel_name', 'pair_name', 'chromophore', ...
    'target_status', 'data_row_kind', 'curve_source', 'time_s', 'signal', 'signal_se', ...
    'task_regressor', 'task_regressor_order', 'is_primary_task_regressor', ...
    'beta', 'se', 't_value', 'p_value', 'dfe', 'shift_index', 'shift_s'};
end

function col = local_find_column(tbl, candidates)
col = '';
vars = lower(string(tbl.Properties.VariableNames));
for i = 1:numel(candidates)
    idx = find(vars == lower(string(candidates{i})), 1, 'first');
    if ~isempty(idx)
        col = tbl.Properties.VariableNames{idx};
        return;
    end
end
end

function out = local_scalar_string(value)
if isstring(value)
    out = char(value(1));
elseif iscell(value)
    out = char(string(value{1}));
elseif ischar(value)
    out = value;
else
    out = char(string(value(1)));
end
end

function out = local_to_cellstr(value)
if isempty(value)
    out = {};
elseif isstring(value)
    out = cellstr(value(:));
elseif ischar(value)
    out = cellstr(value);
elseif iscell(value)
    out = cellfun(@(x) char(string(x)), value(:), 'UniformOutput', false);
else
    out = cellstr(string(value(:)));
end
out = out(~cellfun(@isempty, out));
end
