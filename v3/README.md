Updated MATLAB path for AR-IRLS:
- Replaced direct low-level nirs.math.ar_irls loop with documented module workflow:
  nirs.modules.Resample -> nirs.modules.GLM(type='AR-IRLS')
- MATLAB helper now builds single-channel nirs.core.Data objects with:
  - task stimulus as nirs.design.StimulusEvents
  - nuisance regressors as nirs.design.StimulusVector with regressor_no_interest=true
- Python bundle now includes stimulus timing and per-channel nuisance time series for MATLAB.

Suggested test first:
1. In the notebook, rebuild bundle_path using the updated Python file.
2. Call bench.run_matlab_sidecar_engine(config, bundle_path, job_dir, helper_m)
3. Inspect:
   - matlab_engine_preflight.log
   - matlab_engine_output.log
   - output CSV files in job_dir/matlab_inputs
