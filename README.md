# Twin-CPEE SMC Reliability Verification

This repository contains the code to verify reliability through SMC as presented for the evaluation of the paper 'A Service-Oriented Digital Twin Architecture for Seamless Reliability Verification of IoT Systems'.
The necessary time-series of sensor variables per run and per IoT system with initial parameters is stored in results/results.json.
The *results.json* is used by the script *smc_sim_data.py* to verify reliability for each IoT system with its respective initial parameter.
Hence, a quick way of reproducing the reliability verification in the evaluation is to execute the  *smc_sim_data.py* script.

## Getting started
Tested with: Python 3.11, Linux (Fedora)


1. Execute install.sh 
   - Prepares python environment, installs dependencies, downloads dataset from Zenodo, unpacks the dataset
2. Execute extract_variable_timeseries.py
   - Reads each run, extracts the relevant sensor variables with timestamps, and writes the time series to results.json
3. Execute smc_sim_data.py
   - Reads results.json and verifies reliability with sound and efficient SMC using fixed-k and sequential SMC and the Clopper-Pearson interval
