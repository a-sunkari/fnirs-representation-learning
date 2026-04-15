# fNIRS Representation Learning and Semisynthetic Pipeline Benchmark

This repository contains code for three connected parts of the project:

1. building and organizing semisynthetic fNIRS benchmark inputs  
2. running a multi-pipeline benchmark on those inputs  
3. analyzing benchmark outputs and training a downstream representation-learning model  

---

## Repository contents

The public repository currently includes:

### `cleaned_pipeline_scripts/`
Main benchmark runner and MATLAB helper files.

- `python_runner.py`
- `analyzir_glm_native_batch_v7.m`
- `homer3_tcca_glm_batch_repaired_v2.m`

These files are used to run the benchmark pipelines. The `.m` files should remain in the same directory as `python_runner.py`.

### `pipeline_analysis/`
Post-run analysis code.

- `pipeline_analysis_notebook_v17_clean.py`
- `pipeline_analysis_notebook_v17_clean.ipynb`

Use the `.py` version for script-based figure/export generation and the notebook for interactive exploration.

### `representation_learning/`
Representation-learning training code.

- `train_fnirs_representation_hybrid_v7.py`

### `mat_template_generation/`
HRF template-generation code.

- `mat_generator.py`
- `mat_generator.ipynb`

### `multivariate_phase_randomized_surrogates/`
Surrogate-generation code.

- `mprs_exploration.py`
- `mprs_exploration.ipynb`

### `truth_templates/`
Prebuilt official and custom HRF `.mat` templates used by the benchmark.

### Top-level utilities
- `final_manifest.tsv`
- `transfer_surrogates_to_dataset_script.py`
- `merge_fragmented_fnirs_aggregates_v7.py`
- `environment.yml`

### `synthetic_hrf_generation/`
This folder will also be included with the modified synthetic-injection files used in this project.

Users who want to reconstruct the semisynthetic dataset from scratch should start from the original downloaded source code, then replace the original files with the modified versions provided here and add the extra helper files included in this repository’s `synthetic_hrf_generation/` folder.

---

## What is not included

The public repository does **not** include the benchmark dataset directory itself. The benchmark runner expects a local dataset layout like:

```text
fnirs-representation-learning/
  snirf_dataset_2/
    Subj86/
      resting_clean.snirf
      resting_hrf_20.snirf
      ...
    Subj91/
      ...
````

The public repository also does not guarantee that every local scratch script, cluster submission wrapper, or intermediate output used during development is tracked. The code here is intended to support replication of the benchmark workflow, not to mirror every file in the development environment.

---

## External data and source code needed for replication

To reconstruct the benchmark inputs, download the source data/code from Von Lühmann et al., **“Open Access Multimodal fNIRS Resting State Dataset With and Without Synthetic Hemodynamic Responses.”**

The dataset is available at the following link at [NITRC](https://www.nitrc.org/frs/?group_id=1071)

Required downloads:

* `resting_state_2.zip`
* `code.zip`

After downloading:

1. extract `resting_state_2.zip` and rename the extracted folder to `snirf_dataset_2`
2. extract `code.zip` and rename the extracted folder to `synthetic_hrf_generation`
3. place both folders in the repository root
4. replace the original files in `synthetic_hrf_generation/` with the modified files provided in this repository
5. add the extra helper files included in this repository’s `synthetic_hrf_generation/` folder

So the repository root should look like:

```text
fnirs-representation-learning/
  cleaned_pipeline_scripts/
  pipeline_analysis/
  representation_learning/
  mat_template_generation/
  multivariate_phase_randomized_surrogates/
  synthetic_hrf_generation/
  truth_templates/
  snirf_dataset_2/
  final_manifest.tsv
  environment.yml
  ...
```

---

## Environment setup

Create the base Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate <env-name>
```

### PyTorch install

PyTorch should then be installed with `pip`.

For Linux or Windows with CUDA 12.6:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

For macOS:

```bash
pip3 install torch torchvision
```

### External MATLAB dependencies

Some benchmark pipelines require external MATLAB toolboxes:

* **nirs-toolbox / AnalyzIR**
* **Homer3**

Configure these locally before running MATLAB-backed pipelines.

For the example paths used in this project:

* AnalyzIR: `~/nirs-toolbox`
* Homer3: `~/tools/Homer3`

---

## Replication overview

There are four main workflow stages:

1. prepare HRF templates and semisynthetic benchmark inputs
2. run the benchmark pipelines
3. run the benchmark analysis
4. optionally train the representation-learning model

Each section below lists inputs, outputs, and how to run the code.

---

## 1) HRF templates and semisynthetic dataset preparation

### A. HRF template generation

**Files**

* `mat_template_generation/mat_generator.py`
* `truth_templates/*.mat`

**Purpose**

* generate or inspect official and custom HRF templates used by the benchmark

**Inputs**

* template-generation settings in `mat_generator.py`
* base HRF template information used by the script

**Outputs**

* `.mat` HRF templates
* diagnostic plots comparing templates

**Run**

```bash
python mat_template_generation/mat_generator.py
```

If you are using the prebuilt templates already included in `truth_templates/`, this step is optional.

---

### B. Multivariate phase-randomized surrogate generation

**Files**

* `multivariate_phase_randomized_surrogates/mprs_exploration.py`

**Purpose**

* generate or inspect multivariate phase-randomized surrogate resting-state data used in the benchmark workflow

**Inputs**

* existing resting-state data or arrays expected by the script

**Outputs**

* surrogate data products
* diagnostic plots, depending on script settings

**Run**

```bash
python multivariate_phase_randomized_surrogates/mprs_exploration.py
```

---

### C. Synthetic injection and dataset assembly

**Files**

* `synthetic_hrf_generation/` contents
* `transfer_surrogates_to_dataset_script.py`

**Purpose**

* inject synthetic HRFs into resting-state data
* write benchmark-compatible files
* prepare the final dataset layout expected by the benchmark runner

**Inputs**

* original source code from the downloaded `code.zip`
* modified replacement files in this repository’s `synthetic_hrf_generation/`
* original resting-state dataset in `snirf_dataset_2/`

**Outputs**

* semisynthetic SNIRF files placed into the subject folders under `snirf_dataset_2/`

**Important note**
To reconstruct the semisynthetic dataset from scratch, use the original downloaded code folder as the base, then replace/add the project-specific modified files from this repository’s `synthetic_hrf_generation/` folder.

These files cover tasks such as:

* synthetic HRF injection
* writing benchmark-compatible files
* handling cases where stimulus structure is missing or needs to be added
* preparing outputs for placement into the final dataset layout

After generation, newly created files can be moved into the expected subject-folder dataset structure using:

```bash
python transfer_surrogates_to_dataset_script.py
```

Use this utility only if you are reconstructing or repairing the dataset layout.

---

## 2) Benchmark pipeline execution

### Main benchmark runner

**Files**

* `cleaned_pipeline_scripts/python_runner.py`
* `cleaned_pipeline_scripts/analyzir_glm_native_batch_v7.m`
* `cleaned_pipeline_scripts/homer3_tcca_glm_batch_repaired_v2.m`
* `final_manifest.tsv`

**Purpose**

* run the semisynthetic benchmark across all intended subject/file jobs
* write per-job benchmark outputs
* aggregate completed outputs into a final `aggregate/` directory

**Inputs**

* prepared dataset root containing `snirf_dataset_2/SubjXX/...`
* `truth_templates/`
* `final_manifest.tsv`
* optional external MATLAB dependencies:

  * AnalyzIR / nirs-toolbox
  * Homer3

**Outputs**
Per-job output directories containing tables such as:

* `channel_availability`
* `canonical_channel_metrics`
* `block_average_channel_metrics`
* `fir_channel_metrics`
* `shape_fidelity`
* `roi_timecourses`
* `nuisance_detail`
* `empirical_null_shift`

After aggregation, a final `aggregate/` directory is produced.

### Important replication note

To replicate the benchmark experiment, the intended manifest should be treated as the authoritative job list. In practice, the full experiment means running **all subject/file jobs represented in `final_manifest.tsv`**, not just a single manual runner call.

The normal replication pattern is:

1. prepare the full dataset
2. use the manifest to launch all intended jobs
3. write per-job outputs
4. aggregate the completed outputs once all jobs are done

### Local single-job example

```bash
python cleaned_pipeline_scripts/python_runner.py \
  --subject Subj86 \
  --filename resting_hrf_20.snirf \
  --output-dirname benchmark_outputs_example
```

### Aggregate-only step

```bash
python cleaned_pipeline_scripts/python_runner.py \
  --output-dirname benchmark_outputs_example \
  --aggregate-only
```

### Useful runner options

The runner supports options including:

* `--subject`
* `--filename`
* `--pipeline-label`
* `--pipeline-labels`
* `--output-dirname`
* `--overwrite`
* `--skip-aggregate`
* `--aggregate-only`

### Important benchmark notes

* `final_manifest.tsv` should be treated as the authoritative job list for the intended experiment.
* The benchmark runner supports both Python-native and native MATLAB-backed pipelines.
* In array/batch execution, it is typical to use `--skip-aggregate` during per-job runs and perform a separate final aggregation step afterward.
* `merge_fragmented_fnirs_aggregates_v7.py` is optional and only needed when multiple partial benchmark runs must be merged.

---

### Optional merge utility (can/should be ignored)

I would ignore this file, it is only provided for transparency. When this project was being completed initially, the pipeline results had to be merged as multiple pipelines were incorrectly used, and several runs had to be completed. This file serves as a template for any future merges that would need to be completed.

**File**

* `merge_fragmented_fnirs_aggregates_v7.py`

**Purpose**

* merge outputs from multiple fragmented benchmark runs into one final aggregate

**Inputs**

* multiple benchmark output directories or aggregate fragments

**Outputs**

* one merged aggregate directory

This is not part of the normal single-run replication path.

---

## 3) Running on Cheaha (Slurm)

The benchmark is designed to run well on UAB Cheaha through Slurm array jobs.

### Recommended settings used for this project

* **8 CPU cores per task**
* **32 GB RAM per task**
* **33 concurrent jobs at a time**
* one array task per manifest row
* aggregate after all array tasks finish

### Cheaha execution pattern

The intended pattern is:

1. use `final_manifest.tsv` as the experiment job list
2. submit one array task per manifest row
3. in each array task, read the selected manifest row and extract the subject and filename
4. call the Python runner for that one subject/file job
5. use `--skip-aggregate` during the array run
6. run a separate final aggregation step after all array jobs complete

Your attached Cheaha wrapper follows exactly this model: it reads one manifest line, parses `SUBJECT` and `FILENAME`, and calls the runner with `--skip-aggregate`, `--overwrite`, and `--output-dirname`. 

### Example Slurm header

```bash
#!/bin/bash
#SBATCH --job-name=fnirs_benchmark
#SBATCH --array=0-<N-1>%33
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
```

### Example per-task command

```bash
python cleaned_pipeline_scripts/python_runner.py \
  --subject "$SUBJECT" \
  --filename "$FILENAME" \
  --output-dirname "$OUTNAME" \
  --overwrite \
  --skip-aggregate
```

### Final aggregation command

After the full array completes:

```bash
python cleaned_pipeline_scripts/python_runner.py \
  --output-dirname "$OUTNAME" \
  --aggregate-only
```

### Notes

* For true experiment replication, all intended manifest rows should be completed before the final aggregation step.
* A local/site-specific `.sh` or `sbatch` wrapper may still be needed depending on your cluster setup.

---

## 4) Benchmark analysis

**Files**

* `pipeline_analysis/pipeline_analysis_notebook_v17_clean.py`
* `pipeline_analysis/pipeline_analysis_notebook_v17_clean.ipynb`

**Purpose**

* load aggregate benchmark outputs
* generate figures and exported summary tables
* compare pipelines on recovery, waveform fidelity, amplitude behavior, null behavior, retention, and DLPFC bridge summaries

**Inputs**

* aggregated benchmark outputs from the benchmark runner
* truth templates in `truth_templates/`

**Outputs**

* exported `.csv` analysis tables
* analysis figures

### Run as a script

```bash
python pipeline_analysis/pipeline_analysis_notebook_v17_clean.py
```

### Run as a notebook

Open:

* `pipeline_analysis/pipeline_analysis_notebook_v17_clean.ipynb`

in Jupyter or VS Code.

### Important analysis note

The cleaned analysis notebook/script keeps an `EXCLUDED_PIPELINES` filter so that intentionally excluded or legacy pipelines can be removed from analysis without modifying the aggregate outputs themselves. This was implemented during development when there were several pipelines which were improperly implemented and replaced with newer versions, and had to be removed.

---

## 5) Representation learning

**File**

* `representation_learning/train_fnirs_representation_hybrid_v7.py`

**Purpose**

* train the downstream representation-learning model on benchmark-derived views and features

**Inputs**

* processed benchmark outputs prepared for learning
* model/training settings defined in the script or passed as arguments

**Outputs**

* trained model checkpoints
* logs and training metrics
* evaluation artifacts, depending on script settings

**Run**

```bash
python representation_learning/train_fnirs_representation_hybrid_v7.py
```

---

## Expected inputs and outputs

### Inputs required to replicate the benchmark and analysis

* this repository
* a prepared `snirf_dataset_2/` directory
* `final_manifest.tsv`
* `truth_templates/*.mat`
* a working Conda environment from `environment.yml`
* PyTorch installed for your platform
* MATLAB + external toolboxes if native AnalyzIR/Homer3 pipelines are enabled

### Main outputs

* benchmark per-job output directories
* aggregated benchmark tables
* exported analysis tables
* analysis figures
* optional trained representation-learning model artifacts

---

## Minimal replication path

For someone who wants to reproduce the benchmark and analysis without rebuilding the semisynthetic dataset from scratch (only using the provided 20, 50, and 100 HRF files provided by the dataset instead of the custom HRFs):

1. create the environment from `environment.yml`
2. install PyTorch for your platform
3. prepare `snirf_dataset_2/`
4. confirm that `final_manifest.tsv` matches the intended run
5. run all intended benchmark jobs, locally or via Cheaha/Slurm
6. aggregate the outputs
7. run `pipeline_analysis/pipeline_analysis_notebook_v17_clean.py`

If desired, then run the representation-learning script afterward.
