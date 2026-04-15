function homer3_tcca_glm_batch_repaired_v2()
% Native Homer3 tCCA + GLM batch helper used by the Python benchmark runner.
% It reads one JSON spec, runs the Homer3-side processing, and exports flat
% tables for ROI curves, channel curves, beta values, and metadata.
spec_json = getenv('FNIRS_HOMER3_SPEC_JSON');
if isempty(spec_json)
    error('FNIRS_HOMER3_SPEC_JSON environment variable was not set.');
end

homer3_path = getenv('HOMER3_PATH');
if isempty(homer3_path)
    homer3_path = getenv('FNIRS_HOMER3_PATH');
end
if isempty(homer3_path)
    error('Neither HOMER3_PATH nor FNIRS_HOMER3_PATH environment variable was set.');
end
if exist(homer3_path, 'dir') ~= 7
    error('Resolved Homer3 path does not exist: %s', homer3_path);
end

addpath(genpath(homer3_path));
if exist(fullfile(homer3_path, 'setpaths.m'), 'file') == 2
    old_pwd = pwd;
    cd(homer3_path);
    try
        setpaths;
    catch ME
        warning('setpaths failed: %s', ME.message);
    end
    cd(old_pwd);
end

if isempty(which('hmrR_tCCA'))
    error('hmrR_tCCA was not found on the MATLAB path.');
end
if isempty(which('hmrR_GLM'))
    error('hmrR_GLM was not found on the MATLAB path.');
end
if isempty(which('SnirfClass'))
    error('SnirfClass was not found on the MATLAB path.');
end
if isempty(which('StimClass'))
    error('StimClass was not found on the MATLAB path.');
end

cfg = jsondecode(fileread(spec_json));

active_file = local_scalar_string(cfg.active_snirf_file);
rest_file = local_scalar_string(cfg.resting_snirf_file);
output_roi_csv = local_get_string_field(cfg, 'output_roi_csv_file', '');
if isempty(output_roi_csv) && isfield(cfg, 'output_csv_file')
    output_roi_csv = local_scalar_string(cfg.output_csv_file);
end
output_channel_csv = local_get_string_field(cfg, 'output_channel_csv_file', '');
output_beta_csv = local_get_string_field(cfg, 'output_beta_csv_file', '');
output_metadata_csv = local_get_string_field(cfg, 'output_metadata_csv_file', '');

subject = local_scalar_string(cfg.subject);
file_label = local_scalar_string(cfg.file_label);
pipeline_label = local_scalar_string(cfg.pipeline_label);
backend = local_scalar_string(cfg.backend);
hrf_model = local_scalar_string(cfg.hrf_model);
solver = local_scalar_string(cfg.solver);
amplitude_value = double(cfg.amplitude_value);

target_pair_names = local_to_cellstr(cfg.target_pair_names);
short_sep_thresh = local_get_field_or_default(cfg, 'short_separation_threshold_mm', 15.0);
long_sep_thresh = local_get_field_or_default(cfg, 'long_separation_threshold_mm', 25.0);
tcca_params = local_get_vector_or_default(cfg, 'tcca_params', [3.0, 0.08, 10.0]);
t_rest = local_get_vector_or_default(cfg, 'tcca_rest_window_s', [30.0, 210.0]);
trange = local_get_vector_or_default(cfg, 'glm_trange_s', [-2.0, 17.0]);
ppf = local_get_vector_or_default(cfg, 'ppf', [6.0, 6.0]);
if numel(ppf) == 1
    ppf = [ppf, ppf];
end
lowpass_hz = local_get_field_or_default(cfg, 'lowpass_hz', 0.5);
aux_label_allowlist = local_get_cellstr_field_or_default(cfg, 'aux_label_allowlist', {'acc', 'ppg', 'bp', 'resp'});
ss_channel_selection = local_get_vector_or_default(cfg, 'ss_channel_selection', 0);
local_ensure_parent_dir(output_roi_csv);
local_ensure_parent_dir(output_channel_csv);
local_ensure_parent_dir(output_beta_csv);
local_ensure_parent_dir(output_metadata_csv);

active = SnirfClass(active_file);
resting = SnirfClass(rest_file);
stim = active.stim;
stim_duration_s_effective = local_effective_stim_duration(stim);

if isempty(stim)
    warning('Active SNIRF contains no usable stimulus events. Writing empty outputs.');
    local_write_table(local_empty_roi_table(), output_roi_csv);
    local_write_table(local_empty_channel_hrf_table(), output_channel_csv);
    local_write_table(local_empty_beta_table(), output_beta_csv);
    local_write_table(local_metadata_table(subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, NaN, jsonencode([]), NaN, jsonencode(tcca_params), jsonencode(t_rest), jsonencode(trange), jsonencode(ppf), jsonencode([]), jsonencode([]), double(lowpass_hz), string('no_stim'), 0, 0, 0, 0), output_metadata_csv);
    return;
end

active_dod = hmrR_Intensity2OD(active.data.copy());
rest_dod = hmrR_Intensity2OD(resting.data.copy());

active_dod = hmrR_BandpassFilt(active_dod, 0.0, lowpass_hz);
rest_dod = hmrR_BandpassFilt(rest_dod, 0.0, lowpass_hz);

active_aux = local_get_aux(active);
rest_aux = local_get_aux(resting);
if ~isempty(active_aux)
    active_aux = hmrR_BandpassFilt(active_aux, 0.0, lowpass_hz);
end
if ~isempty(rest_aux)
    rest_aux = hmrR_BandpassFilt(rest_aux, 0.0, lowpass_hz);
end

if exist('hmrR_OD2Conc_new', 'file') == 2
    active_dc = hmrR_OD2Conc_new(active_dod, active.probe, ppf);
    rest_dc = hmrR_OD2Conc_new(rest_dod, resting.probe, ppf);
else
    active_dc = hmrR_OD2Conc(active_dod, active.probe, ppf);
    rest_dc = hmrR_OD2Conc(rest_dod, resting.probe, ppf);
end

rest_aux_matrix = local_get_aux_matrix(resting);
active_aux_matrix = local_get_aux_matrix(active);
rest_time = local_extract_time(rest_dc);
active_time = local_extract_time(active_dc);
shared_aux_indices = local_shared_aux_indices(resting, active);
preferred_aux_indices = local_filter_aux_indices_by_names(shared_aux_indices, resting, active, aux_label_allowlist);
aux_candidate_sets = local_make_aux_candidate_sets(preferred_aux_indices, shared_aux_indices);
ss_candidate_sets = local_make_ss_candidate_sets(ss_channel_selection);
[Aaux, rcMap, aux_indices_used, ss_ch_inx_used, tcca_fallback_note] = local_run_tcca_with_fallbacks(rest_dc, rest_aux, resting.probe, rest_aux_matrix, rest_time, active_dc, active_aux, active.probe, active_aux_matrix, active_time, tcca_params, t_rest, short_sep_thresh, aux_candidate_sets, ss_candidate_sets);

[idx_basis, params_basis] = local_basis_settings(hrf_model);
[dcAvg, ~, ~, dcNew, dcResid, ~, beta_out, ~, hmrstats] = hmrR_GLM(active_dc, stim, active.probe, [], Aaux, [], rcMap, trange, 1, idx_basis, params_basis, short_sep_thresh, 3, 3, []); %#ok<NASGU>

beta_arr = local_extract_first_block(beta_out);

t_avg = local_extract_time(dcAvg);
y_avg = local_extract_timeseries(dcAvg);
ml = local_extract_measlist(dcAvg);
if isempty(ml)
    ml = active_dc.GetMeasListSrcDetPairs('reshape');
end

src_pos = active.probe.GetSrcPos();
det_pos = active.probe.GetDetPos();

n_channels = size(ml, 1);
pair_names = cell(n_channels, 1);
channel_names = cell(n_channels, 2);
is_long = false(n_channels, 1);
is_target = false(n_channels, 1);
rho_mm = nan(n_channels, 1);
for ii = 1:n_channels
    pair_names{ii} = sprintf('S%d_D%d', ml(ii, 1), ml(ii, 2));
    channel_names{ii, 1} = sprintf('%s hbo', pair_names{ii});
    channel_names{ii, 2} = sprintf('%s hbr', pair_names{ii});
    rho_mm(ii) = sqrt(sum((src_pos(ml(ii, 1), :) - det_pos(ml(ii, 2), :)).^2));
    is_long(ii) = rho_mm(ii) > long_sep_thresh;
    is_target(ii) = any(strcmp(pair_names{ii}, target_pair_names));
end
is_target = is_target & is_long;
is_non_target = (~is_target) & is_long;

channel_table = local_make_channel_hrf_table(subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, t_avg, y_avg, channel_names, pair_names, is_long, is_target, idx_basis, params_basis, rho_mm);
roi_table = local_make_roi_table(subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, t_avg, y_avg, is_target, is_non_target, idx_basis, params_basis);
beta_table = local_make_beta_table(subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, beta_arr, channel_names, pair_names, is_long, is_target, idx_basis, params_basis, rho_mm);
metadata_table = local_metadata_table(subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, idx_basis, jsonencode(params_basis), stim_duration_s_effective, jsonencode(tcca_params), jsonencode(t_rest), jsonencode(trange), jsonencode(ppf), jsonencode(aux_indices_used), jsonencode(ss_ch_inx_used), double(lowpass_hz), string(tcca_fallback_note), numel(aux_indices_used), sum(is_long), sum(is_target), sum(is_non_target));

local_write_table(roi_table, output_roi_csv);
local_write_table(channel_table, output_channel_csv);
local_write_table(beta_table, output_beta_csv);
local_write_table(metadata_table, output_metadata_csv);
end

function stim_duration_s_effective = local_effective_stim_duration(stim)
stim_duration_s_effective = NaN;
if isempty(stim)
    return;
end
try
    durations = double(stim(1).data(:, 2));
    durations = durations(isfinite(durations) & durations > 0);
    if ~isempty(durations)
        stim_duration_s_effective = median(durations);
    end
catch
    stim_duration_s_effective = NaN;
end
end

function beta_arr = local_extract_first_block(beta_out)
beta_arr = [];
if isempty(beta_out)
    return;
end
if iscell(beta_out)
    if isempty(beta_out{1})
        return;
    end
    beta_arr = beta_out{1};
else
    beta_arr = beta_out;
end
end

function t = local_extract_time(data_obj)
if numel(data_obj) > 1
    data_obj = data_obj(1);
end
t = double(data_obj.time(:));
end

function y = local_extract_timeseries(data_obj)
if numel(data_obj) > 1
    data_obj = data_obj(1);
end
y = data_obj.GetDataTimeSeries('reshape');
y = double(y);
end

function ml = local_extract_measlist(data_obj)
ml = [];
try
    if numel(data_obj) > 1
        data_obj = data_obj(1);
    end
    ml = data_obj.GetMeasListSrcDetPairs('reshape');
catch
    ml = [];
end
end

function sig = local_extract_curve(y_avg, chrom_idx, ch_idx)
if isempty(y_avg)
    sig = [];
    return;
end
if ndims(y_avg) == 2
    sig = squeeze(y_avg(:, ch_idx));
elseif ndims(y_avg) == 3
    sig = squeeze(y_avg(:, chrom_idx, ch_idx));
else
    sig = squeeze(y_avg(:, chrom_idx, ch_idx, 1));
end
sig = double(sig(:));
end

function value = local_extract_beta(beta_arr, beta_idx, chrom_idx, ch_idx)
if isempty(beta_arr)
    value = NaN;
    return;
end
if ndims(beta_arr) == 2
    value = beta_arr(beta_idx, ch_idx);
elseif ndims(beta_arr) == 3
    value = beta_arr(beta_idx, chrom_idx, ch_idx);
else
    value = beta_arr(beta_idx, chrom_idx, ch_idx, 1);
end
value = double(value);
end

function T = local_make_channel_hrf_table(subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, t_avg, y_avg, channel_names, pair_names, is_long, is_target, idx_basis, params_basis, rho_mm)
rows = {};
chrom_names = {'hbo', 'hbr'};
for ch_idx = 1:numel(pair_names)
    if ~is_long(ch_idx)
        continue;
    end
    if is_target(ch_idx)
        target_status = 'true_target';
    else
        target_status = 'true_non_target';
    end
    for chrom_idx = 1:min(2, size(y_avg, 2))
        sig = local_extract_curve(y_avg, chrom_idx, ch_idx);
        if isempty(sig)
            continue;
        end
        for ii = 1:numel(t_avg)
            rows(end + 1, :) = {subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, channel_names{ch_idx, chrom_idx}, pair_names{ch_idx}, chrom_names{chrom_idx}, target_status, 'homer3_dcAvg', double(t_avg(ii)), double(sig(ii)), 1, idx_basis, jsonencode(params_basis), rho_mm(ch_idx)}; %#ok<AGROW>
        end
    end
end
var_names = {'subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'hrf_model', 'solver', 'channel_name', 'pair_name', 'chromophore', 'target_status', 'curve_source', 'time_s', 'signal', 'condition_index', 'idx_basis', 'params_basis_json', 'source_detector_distance_mm'};
if isempty(rows)
    T = cell2table(cell(0, numel(var_names)), 'VariableNames', var_names);
else
    T = cell2table(rows, 'VariableNames', var_names);
end
end

function T = local_make_roi_table(subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, t_avg, y_avg, is_target, is_non_target, idx_basis, params_basis)
rows = {};
chrom_names = {'hbo', 'hbr'};
for chrom_idx = 1:min(2, size(y_avg, 2))
    if any(is_target)
        curves = [];
        idxs = find(is_target(:)');
        for ch_idx = idxs
            curves = [curves, local_extract_curve(y_avg, chrom_idx, ch_idx)]; %#ok<AGROW>
        end
        if ~isempty(curves)
            sig = mean(curves, 2);
            for ii = 1:numel(t_avg)
                rows(end + 1, :) = {subject, file_label, amplitude_value, pipeline_label, backend, chrom_names{chrom_idx}, 'true_target', 'homer3_roi_mean_channel_hrf', double(t_avg(ii)), double(sig(ii)), 1, idx_basis, jsonencode(params_basis)}; %#ok<AGROW>
            end
        end
    end
    if any(is_non_target)
        curves = [];
        idxs = find(is_non_target(:)');
        for ch_idx = idxs
            curves = [curves, local_extract_curve(y_avg, chrom_idx, ch_idx)]; %#ok<AGROW>
        end
        if ~isempty(curves)
            sig = mean(curves, 2);
            for ii = 1:numel(t_avg)
                rows(end + 1, :) = {subject, file_label, amplitude_value, pipeline_label, backend, chrom_names{chrom_idx}, 'true_non_target', 'homer3_roi_mean_channel_hrf', double(t_avg(ii)), double(sig(ii)), 1, idx_basis, jsonencode(params_basis)}; %#ok<AGROW>
            end
        end
    end
end
var_names = {'subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore', 'target_status', 'curve_source', 'time_s', 'signal', 'condition_index', 'idx_basis', 'params_basis_json'};
if isempty(rows)
    T = cell2table(cell(0, numel(var_names)), 'VariableNames', var_names);
else
    T = cell2table(rows, 'VariableNames', var_names);
end
end

function T = local_make_beta_table(subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, beta_arr, channel_names, pair_names, is_long, is_target, idx_basis, params_basis, rho_mm)
var_names = {'subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'hrf_model', 'solver', 'channel_name', 'pair_name', 'chromophore', 'target_status', 'beta_index', 'beta_value', 'condition_index', 'idx_basis', 'params_basis_json', 'source_detector_distance_mm'};
if isempty(beta_arr)
    T = cell2table(cell(0, numel(var_names)), 'VariableNames', var_names);
    return;
end
rows = {};
chrom_names = {'hbo', 'hbr'};
n_beta = size(beta_arr, 1);
for ch_idx = 1:numel(pair_names)
    if ~is_long(ch_idx)
        continue;
    end
    if is_target(ch_idx)
        target_status = 'true_target';
    else
        target_status = 'true_non_target';
    end
    for chrom_idx = 1:min(2, size(beta_arr, 2))
        for beta_idx = 1:n_beta
            beta_value = local_extract_beta(beta_arr, beta_idx, chrom_idx, ch_idx);
            rows(end + 1, :) = {subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, channel_names{ch_idx, chrom_idx}, pair_names{ch_idx}, chrom_names{chrom_idx}, target_status, beta_idx, beta_value, 1, idx_basis, jsonencode(params_basis), rho_mm(ch_idx)}; %#ok<AGROW>
        end
    end
end
if isempty(rows)
    T = cell2table(cell(0, numel(var_names)), 'VariableNames', var_names);
else
    T = cell2table(rows, 'VariableNames', var_names);
end
end

function T = local_metadata_table(subject, file_label, amplitude_value, pipeline_label, backend, hrf_model, solver, idx_basis, params_basis_json, stim_duration_s_effective, tcca_params_json, tcca_rest_window_json, trange_json, ppf_json, aux_indices_json, ss_ch_inx_json, lowpass_hz, tcca_fallback_note, n_aux_indices, n_long_channels, n_target_long_channels, n_non_target_long_channels)
T = table( ...
    string(subject), string(file_label), double(amplitude_value), string(pipeline_label), string(backend), string(hrf_model), string(solver), ...
    double(idx_basis), string(params_basis_json), double(stim_duration_s_effective), string(tcca_params_json), string(tcca_rest_window_json), string(trange_json), string(ppf_json), string(aux_indices_json), string(ss_ch_inx_json), double(lowpass_hz), string(tcca_fallback_note), ...
    double(n_aux_indices), double(n_long_channels), double(n_target_long_channels), double(n_non_target_long_channels), ...
    'VariableNames', {'subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'hrf_model', 'solver', 'idx_basis', 'params_basis_json', 'stim_duration_s_effective', 'tcca_params_json', 'tcca_rest_window_s_json', 'glm_trange_s_json', 'ppf_json', 'aux_indices_json', 'ss_ch_inx_json', 'lowpass_hz', 'tcca_fallback_note', 'n_aux_indices', 'n_long_channels', 'n_target_long_channels', 'n_non_target_long_channels'} ...
);
end

function T = local_empty_roi_table()
T = cell2table(cell(0, 13), 'VariableNames', {'subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore', 'target_status', 'curve_source', 'time_s', 'signal', 'condition_index', 'idx_basis', 'params_basis_json'});
end

function T = local_empty_channel_hrf_table()
T = cell2table(cell(0, 18), 'VariableNames', {'subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'hrf_model', 'solver', 'channel_name', 'pair_name', 'chromophore', 'target_status', 'curve_source', 'time_s', 'signal', 'condition_index', 'idx_basis', 'params_basis_json', 'source_detector_distance_mm'});
end

function T = local_empty_beta_table()
T = cell2table(cell(0, 17), 'VariableNames', {'subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'hrf_model', 'solver', 'channel_name', 'pair_name', 'chromophore', 'target_status', 'beta_index', 'beta_value', 'condition_index', 'idx_basis', 'params_basis_json', 'source_detector_distance_mm'});
end

function aux = local_get_aux(snirf_obj)
aux = [];
try
    if isprop(snirf_obj, 'aux')
        aux = snirf_obj.aux;
    end
catch
    aux = [];
end
end



function aux_matrix = local_get_aux_matrix(snirf_obj)
aux_matrix = [];
try
    s = SnirfClass();
    s.data = snirf_obj.data;
    s.aux = snirf_obj.aux;
    aux_matrix = s.GetAuxDataMatrix();
catch
    try
        aux_matrix = snirf_obj.GetAuxDataMatrix();
    catch
        aux_matrix = [];
    end
end
end

function aux_indices = local_shared_aux_indices(rest_snirf, active_snirf)
rest_matrix = local_get_aux_matrix(rest_snirf);
active_matrix = local_get_aux_matrix(active_snirf);
if isempty(rest_matrix) || isempty(active_matrix)
    aux_indices = [];
    return;
end
n_common = min(size(rest_matrix, 2), size(active_matrix, 2));
if n_common <= 0
    aux_indices = [];
    return;
end
aux_indices = 1:n_common;
end

function aux_indices = local_filter_aux_indices_by_names(shared_aux_indices, rest_snirf, active_snirf, allowlist)
if isempty(shared_aux_indices)
    aux_indices = [];
    return;
end
rest_names = local_get_aux_names(rest_snirf);
active_names = local_get_aux_names(active_snirf);
keep = false(size(shared_aux_indices));
for ii = 1:numel(shared_aux_indices)
    idx = shared_aux_indices(ii);
    aux_name = local_choose_aux_name(rest_names, active_names, idx);
    keep(ii) = local_name_matches_allowlist(aux_name, allowlist);
end
aux_indices = shared_aux_indices(keep);
end

function names = local_get_aux_names(snirf_obj)
names = {};
try
    names = snirf_obj.GetAuxNames();
catch
    names = {};
end
if isempty(names)
    names = {};
end
end

function aux_name = local_choose_aux_name(rest_names, active_names, idx)
aux_name = '';
if numel(rest_names) >= idx
    aux_name = local_scalar_string(rest_names{idx});
end
if isempty(strtrim(aux_name)) && numel(active_names) >= idx
    aux_name = local_scalar_string(active_names{idx});
end
aux_name = lower(strtrim(aux_name));
end

function tf = local_name_matches_allowlist(aux_name, allowlist)
tf = false;
if isempty(aux_name)
    return;
end
for ii = 1:numel(allowlist)
    token = lower(strtrim(local_scalar_string(allowlist{ii})));
    if ~isempty(token) && contains(aux_name, token)
        tf = true;
        return;
    end
end
end

function candidate_sets = local_make_aux_candidate_sets(preferred_aux_indices, shared_aux_indices)
candidate_sets = {};
if ~isempty(preferred_aux_indices)
    candidate_sets{end+1} = preferred_aux_indices(:)'; %#ok<AGROW>
end
if ~isempty(shared_aux_indices)
    candidate_sets{end+1} = shared_aux_indices(:)'; %#ok<AGROW>
end
candidate_sets{end+1} = []; %#ok<AGROW>
candidate_sets = local_unique_numeric_row_sets(candidate_sets);
end

function candidate_sets = local_make_ss_candidate_sets(ss_channel_selection)
candidate_sets = {};
if ~isempty(ss_channel_selection)
    candidate_sets{end+1} = double(ss_channel_selection(:)'); %#ok<AGROW>
end
candidate_sets{end+1} = 0; %#ok<AGROW>
candidate_sets{end+1} = 1; %#ok<AGROW>
candidate_sets = local_unique_numeric_row_sets(candidate_sets);
end

function out = local_unique_numeric_row_sets(cell_sets)
out = {};
keys = {};
for ii = 1:numel(cell_sets)
    value = double(cell_sets{ii}(:)');
    if isempty(value)
        key = '[]';
    else
        key = sprintf('%g,', value);
    end
    if ~any(strcmp(keys, key))
        out{end+1} = value; %#ok<AGROW>
        keys{end+1} = key; %#ok<AGROW>
    end
end
end

function [Aaux, rcMap, aux_indices_used, ss_ch_inx_used, fallback_note] = local_run_tcca_with_fallbacks(rest_dc, rest_aux, rest_probe, rest_aux_matrix, rest_time, active_dc, active_aux, active_probe, active_aux_matrix, active_time, tcca_params, t_rest, short_sep_thresh, aux_candidate_sets, ss_candidate_sets)
Aaux = [];
rcMap = [];
aux_indices_used = [];
ss_ch_inx_used = [];
fallback_note = 'uninitialized';
messages = {};
for iAux = 1:numel(aux_candidate_sets)
    candidate_aux = double(aux_candidate_sets{iAux}(:)');
    candidate_aux = local_validate_aux_candidate(candidate_aux, rest_aux_matrix, rest_time, t_rest, active_aux_matrix, active_time);
    for iSS = 1:numel(ss_candidate_sets)
        candidate_ss = double(ss_candidate_sets{iSS}(:)');
        if isempty(candidate_ss)
            candidate_ss = 0;
        end
        try
            [~, ~] = hmrR_tCCA(rest_dc, rest_aux, rest_probe, 1, 1, [], [], 1, tcca_params, candidate_aux, short_sep_thresh, candidate_ss, 1, t_rest);
            [Aaux, rcMap] = hmrR_tCCA(active_dc, active_aux, active_probe, 2, 1, [], [], 1, tcca_params, candidate_aux, short_sep_thresh, candidate_ss, 1, t_rest);
            aux_indices_used = candidate_aux;
            ss_ch_inx_used = candidate_ss;
            fallback_note = sprintf('aux_try_%d_ss_try_%d', iAux, iSS);
            return;
        catch ME
            messages{end+1} = sprintf('aux=%s ss=%s :: %s', mat2str(candidate_aux), mat2str(candidate_ss), ME.message); %#ok<AGROW>
        end
    end
end
error('All tCCA fallback attempts failed. %s', strjoin(messages, ' || '));
end

function aux_indices = local_validate_aux_candidate(candidate_aux, rest_aux_matrix, rest_time, t_rest, active_aux_matrix, active_time)
aux_indices = [];
if isempty(candidate_aux)
    return;
end
rest_mask = true(size(rest_time));
if ~isempty(rest_time) && numel(t_rest) >= 2
    rest_mask = rest_time >= t_rest(1) & rest_time <= t_rest(2);
    if ~any(rest_mask)
        rest_mask = true(size(rest_time));
    end
end
active_mask = true(size(active_time));
for ii = 1:numel(candidate_aux)
    idx = candidate_aux(ii);
    if idx < 1 || idx > size(rest_aux_matrix, 2) || idx > size(active_aux_matrix, 2)
        continue;
    end
    rest_col = double(rest_aux_matrix(:, idx));
    active_col = double(active_aux_matrix(:, idx));
    rest_col = rest_col(rest_mask);
    active_col = active_col(active_mask);
    rest_col = rest_col(isfinite(rest_col));
    active_col = active_col(isfinite(active_col));
    if numel(rest_col) > 1 && numel(active_col) > 1 && std(rest_col) > 0 && std(active_col) > 0
        aux_indices(end+1) = idx; %#ok<AGROW>
    end
end
end

function [idx_basis, params_basis] = local_basis_settings(hrf_model)
model = lower(char(hrf_model));
switch model
    case 'homer3_gaussian'
        idx_basis = 1;
        params_basis = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0];
    case 'homer3_modgamma'
        idx_basis = 2;
        params_basis = [0.1, 3.0, 1.8, 3.0];
    otherwise
        error('Unsupported Homer3 HRF model: %s', model);
end
end

function value = local_get_field_or_default(cfg, field_name, default_value)
if isfield(cfg, field_name)
    value = double(cfg.(field_name));
else
    value = default_value;
end
end

function value = local_get_vector_or_default(cfg, field_name, default_value)
if isfield(cfg, field_name)
    value = double(cfg.(field_name));
else
    value = default_value;
end
value = value(:)';
end

function value = local_get_cellstr_field_or_default(cfg, field_name, default_value)
if isfield(cfg, field_name)
    value = local_to_cellstr(cfg.(field_name));
else
    value = default_value;
end
end

function value = local_get_string_field(cfg, field_name, default_value)
if isfield(cfg, field_name)
    value = local_scalar_string(cfg.(field_name));
else
    value = default_value;
end
end

function value = local_scalar_string(input_value)
if isstring(input_value)
    value = char(input_value);
elseif ischar(input_value)
    value = input_value;
elseif iscell(input_value) && ~isempty(input_value)
    value = local_scalar_string(input_value{1});
else
    value = char(string(input_value));
end
end

function value = local_to_cellstr(input_value)
if isempty(input_value)
    value = {};
    return;
end
if iscell(input_value)
    value = cellfun(@local_scalar_string, input_value, 'UniformOutput', false);
elseif isstring(input_value)
    value = cellstr(input_value);
elseif ischar(input_value)
    value = {input_value};
else
    value = cellstr(string(input_value));
end
end

function local_ensure_parent_dir(filename)
if isempty(filename)
    return;
end
parent_dir = fileparts(filename);
if ~isempty(parent_dir) && exist(parent_dir, 'dir') ~= 7
    mkdir(parent_dir);
end
end

function local_write_table(T, filename)
if isempty(filename)
    return;
end
local_ensure_parent_dir(filename);
writetable(T, filename);
end
