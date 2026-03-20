# %% [markdown]
# # fNIRS benchmark serial smoke-test notebook
# 
# This notebook isolates the first `no_hrf` job **without** `ProcessPoolExecutor` so you can find out whether the failure is in:
# 
# 1. Python preprocessing / GLM,
# 2. MATLAB bundle construction, or
# 3. MATLAB Engine execution.
# 
# Run the cells top to bottom. The notebook only touches **one subject** and a couple of representative pipelines unless you expand it.
# 

# %%

from pathlib import Path
import importlib.util
import traceback
import time
import json

import numpy as np
import pandas as pd
import mne

# --- EDIT THESE PATHS IF NEEDED ---
SCRIPT_PATH = Path('/home/asunkari/fnirs-representation-learning/v1/fnirs_benchmark_v7_post_changes.py')
ROOT = Path('/home/asunkari/fnirs-representation-learning')
TRUTH_TEMPLATE_DIR = ROOT / 'synthetic_hrf_generation'
ANALYZIR_PATH = Path('/home/asunkari/nirs-toolbox')
SUBJECT = 'Subj94'

assert SCRIPT_PATH.exists(), SCRIPT_PATH
assert ROOT.exists(), ROOT
assert ANALYZIR_PATH.exists(), ANALYZIR_PATH
print('Using script:', SCRIPT_PATH)
print('Using root:', ROOT)
print('Using truth dir:', TRUTH_TEMPLATE_DIR)
print('Using AnalyzIR path:', ANALYZIR_PATH)


# %%
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("bench", str(SCRIPT_PATH))
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)

print("Loaded module:", bench.__name__)
print("MATLAB engine available:", bench.optional_import_matlab_engine()[1] is not None)

# %%

# Build a serial config that matches your local smoke test.
config = bench.BenchmarkConfig(
    root=str(ROOT),
    truth_template_dir=str(TRUTH_TEMPLATE_DIR),
    analyzir_path=str(ANALYZIR_PATH),
    use_matlab=True,
    use_matlab_engine=True,
    prefer_matlab_engine=True,
    empirical_null_shift_count=1,
    n_workers=1,
    overwrite=True,
)
config.file_specs = bench.default_file_specs()
config.pipeline_specs = bench.default_pipeline_specs()

pd.DataFrame([bench.asdict(fs) for fs in config.file_specs])


# %%
df_pipes = pd.DataFrame([bench.asdict(ps) for ps in config.pipeline_specs])
df_pipes[['label','backend','nuisance_method','hrf_model','solver','pruning_style','motion_method','filter_mode','use_block_average']]

# %% [markdown]
# ## Helpers
# 
# `prepare_job_state()` reproduces the early steps of `process_subject_file_job()` in the **main notebook process**.
# 

# %%

def prepare_job_state(file_label='no_hrf'):
    file_spec = next(fs for fs in config.file_specs if fs.label == file_label)
    dataset_dir = config.dataset_path()
    subject_dir = dataset_dir / SUBJECT
    snirf_file_path = subject_dir / file_spec.filename
    annotation_source_path = subject_dir / file_spec.annotation_source_filename if file_spec.annotation_source_filename else None
    reference_path = subject_dir / 'resting_hrf_20.snirf'

    if not snirf_file_path.exists():
        raise FileNotFoundError(snirf_file_path)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    print(f'Loading raw CW from: {snirf_file_path}')
    raw_cw = mne.io.read_raw_snirf(snirf_file_path, preload=True, verbose=False)
    if file_spec.is_null:
        print(f'Copying annotations from: {annotation_source_path}')
        raw_cw = bench.copy_valid_annotations(raw_cw, annotation_source_path)
    raw_cw = bench.sanitize_annotations_to_single_task(raw_cw)

    print(f'Loading reference truth file: {reference_path}')
    reference_raw = mne.io.read_raw_snirf(reference_path, preload=True, verbose=False)
    data_type_labels = bench.read_measurement_data_type_labels(reference_path)
    if data_type_labels is None:
        raise RuntimeError('Could not read truth labels from reference file.')

    picks_cw = bench.get_cw_channel_indices(reference_raw)
    cw_names = np.asarray(reference_raw.ch_names)[picks_cw]
    if len(data_type_labels) == len(cw_names):
        aligned_names = cw_names
    elif len(data_type_labels) == len(reference_raw.ch_names):
        aligned_names = np.asarray(reference_raw.ch_names)
    else:
        raise RuntimeError('Truth-label alignment failed.')

    target_pair_names = sorted(set(name.split(' ')[0] for name in aligned_names[data_type_labels == 1].tolist()))
    target_pair_set = set(target_pair_names)

    cw_channel_table = bench.build_cw_channel_table(reference_raw, SUBJECT, file_spec.label, config)
    long_pair_names = sorted(cw_channel_table.loc[cw_channel_table['group'] == 'LS', 'pair_name'].astype(str).unique())
    non_target_pair_names = [pair for pair in long_pair_names if pair not in target_pair_set]

    channel_quality, pair_quality = bench.build_quality_tables(raw_cw, SUBJECT, file_spec.label, config)
    truth_templates = bench.load_truth_templates(config)

    state = {
        'file_spec': file_spec,
        'subject_dir': subject_dir,
        'snirf_file_path': snirf_file_path,
        'raw_cw': raw_cw,
        'reference_raw': reference_raw,
        'target_pair_names': target_pair_names,
        'target_pair_set': target_pair_set,
        'non_target_pair_names': non_target_pair_names,
        'cw_channel_table': cw_channel_table,
        'channel_quality': channel_quality,
        'pair_quality': pair_quality,
        'truth_templates': truth_templates,
    }
    return state


# %%

state = prepare_job_state('no_hrf')
print('Annotations:', len(state['raw_cw'].annotations))
print('True target pairs:', state['target_pair_names'])
print('n true targets:', len(state['target_pair_names']))
print('n true non-target pairs:', len(state['non_target_pair_names']))

state['pair_quality'].head()


# %%

# Quick QC sanity check.
pair_quality = state['pair_quality']
strict_bad = bench.get_bad_pair_names(pair_quality, 'strict_combined', config)
loose_bad = bench.get_bad_pair_names(pair_quality, 'loose_sci', config)

print('Strict bad pairs:', len(strict_bad))
print('Loose bad pairs:', len(loose_bad))
print('Strict good LS pairs:', int((pair_quality['group'].eq('LS') & ~pair_quality['pair_name'].isin(strict_bad)).sum()))
print('Loose good LS pairs:', int((pair_quality['group'].eq('LS') & ~pair_quality['pair_name'].isin(loose_bad)).sum()))


# %% [markdown]
# ## Probe 1: Python-only pipeline on `no_hrf`
# 
# This tests whether the Python preprocessing + GLM path works when run serially in the notebook.
# 

# %%

def run_python_probe(pipeline_label, file_label='no_hrf'):
    local_state = prepare_job_state(file_label)
    file_spec = local_state['file_spec']
    pipeline = next(ps for ps in config.pipeline_specs if ps.label == pipeline_label)
    raw_hb, bad_pairs = bench.preprocess_raw_to_hb(local_state['raw_cw'], local_state['pair_quality'], pipeline, config)
    print('Pipeline:', pipeline.label)
    print('Bad pairs:', len(bad_pairs))
    print('Available long channels after preprocess:', len(bench.get_available_long_channel_names(raw_hb, config)))
    t0 = time.time()
    result = bench.execute_python_pipeline(
        subject=SUBJECT,
        file_spec=file_spec,
        pipeline=pipeline,
        raw_cw=local_state['raw_cw'],
        raw_hb=raw_hb,
        target_pair_names=local_state['target_pair_set'],
        truth_templates=local_state['truth_templates'],
        snirf_file_path=local_state['snirf_file_path'],
        config=config,
    )
    dt = time.time() - t0
    print(f'Finished in {dt:.2f} s')
    print({k: len(v) if hasattr(v, '__len__') else type(v).__name__ for k, v in result.items()})
    return local_state, pipeline, raw_hb, result


# %%

py_state, py_pipeline, py_raw_hb, py_result = run_python_probe('LocalSS_Glover_AUTO', 'no_hrf')


# %%
for key in ['canonical_channel_metrics', 'shape_metrics', 'roi_timecourses', 'nuisance_detail']:
    obj = py_result[key]
    print(f"\n=== {key} ===")
    if isinstance(obj, pd.DataFrame):
        display(obj.head())
    else:
        print(type(obj), len(obj) if hasattr(obj, "__len__") else "")

# %% [markdown]
# ## Probe 2: MATLAB bundle creation on `no_hrf`
# 
# This does **not** launch MATLAB yet. It just verifies that the AR-IRLS branch can build its design/spec bundle in Python.
# 

# %%

mat_state, mat_pipeline, mat_raw_hb, mat_result = run_python_probe('LocalSS_Glover_ARIRLS', 'no_hrf')
print('Observed MATLAB specs:', len(mat_result['matlab_input_specs_list']))
print('Shift MATLAB specs:', len(mat_result['matlab_shift_specs_list']))


# %%

job_dir = config.output_path() / 'notebook_debug' / SUBJECT / 'no_hrf'
job_dir.mkdir(parents=True, exist_ok=True)

bundle_path = bench.write_matlab_bundle(
    mat_result['matlab_input_specs_list'],
    mat_result['matlab_shift_specs_list'],
    job_dir,
)
print('Bundle path:', bundle_path)
print(bundle_path.read_text()[:1000] if bundle_path else 'No bundle written')


# %% [markdown]
# ## Probe 3: MATLAB Engine run in the notebook main process
# 
# This is the crucial test. If this works here but the batch script still dies, the likely issue is **MATLAB Engine inside the worker process**, not the math itself.
# 

# %%
import matlab.engine
eng = matlab.engine.start_matlab("-nojvm")
print(eng.version())

# %%
helper_dir = str(SCRIPT_PATH.parent)
eng.addpath(helper_dir, nargout=0)

eng.setenv("FNIRS_BUNDLE_JSON", str(bundle_path), nargout=0)
eng.setenv("FNIRS_ANALYZIR_PATH", str(config.analyzir_path), nargout=0)

print(eng.eval("which('nirs.modules.AR_IRLS')", nargout=1))
print(eng.eval("which('nirs.math.ar_irls')", nargout=1))
print(eng.eval(f"which('{Path(helper_dir, 'analyzir_arirls_batch.m').stem}')", nargout=1))

# %%
bench.run_matlab_sidecar(config, bundle_path, job_dir)

# %%
"""
matlab_df = None
try:
    t0 = time.time()
    matlab_df = bench.run_matlab_sidecar(config, bundle_path, job_dir)
    dt = time.time() - t0
    print(f"MATLAB sidecar finished in {dt:.2f} s")
    display(matlab_df.head())
    print("Rows:", len(matlab_df))
except Exception as exc:
    print("MATLAB sidecar failed:")
    print(type(exc).__name__, exc)
    traceback.print_exc()

    for candidate in ["matlab_engine.log", "matlab_engine_fallback.log", "matlab_batch.log"]:
        p = job_dir / candidate
        if p.exists():
            print(f"\n--- {candidate} ---")
            print(p.read_text()[:4000])
"""

# %% [markdown]
# ## Optional: run the same representative Python probe on `hrf_20`
# 
# This lets you confirm the active-condition path separately from the null path.
# 

# %%

hrf20_state, hrf20_pipeline, hrf20_raw_hb, hrf20_result = run_python_probe('LocalSS_Glover_AUTO', 'hrf_20')


# %%

if isinstance(hrf20_result['canonical_channel_metrics'], pd.DataFrame):
    display(hrf20_result['canonical_channel_metrics'].head())
if isinstance(hrf20_result['shape_metrics'], pd.DataFrame):
    display(hrf20_result['shape_metrics'].head())


# %% [markdown]
# ## Interpretation guide
# 
# - If **Probe 1 fails**, the issue is in Python preprocessing / GLM before MATLAB.
# - If **Probe 1 works but Probe 2 fails**, the issue is in AR-IRLS spec construction.
# - If **Probe 2 works but Probe 3 fails**, the issue is MATLAB runtime/path/helper behavior.
# - If **Probe 3 works in the notebook but the batch script still dies**, the likely culprit is the **worker-process execution path**, especially MATLAB Engine inside the `ProcessPoolExecutor` worker.
# 


