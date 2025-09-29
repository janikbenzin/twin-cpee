# Twin-CPEE SMC Reliability Verification

This repository contains the code to verify reliability through SMC as presented for the evaluation of the paper 'A Service-Oriented Digital Twin Architecture for Seamless Reliability Verification of IoT Systems'.
The necessary time-series of sensor variables per run and per IoT system with initial parameters is stored in results/results.json.
The *results.json* is used by the script *smc_sim_data.py* to verify reliability for each IoT system with its respective initial parameter.
Hence, a quick way of reproducing the reliability verification in the evaluation is to execute the  *smc_sim_data.py* script.

## Getting started
Tested with: Python 3.11, Linux (Fedora)

### Reproduce evaluation results
1. Execute install.sh 
   - Prepares python environment, installs dependencies, downloads datasets from Zenodo, unpacks datasets
2. Execute extract_variable_timeseries.py
   - Reads each run, extracts the relevant sensor variables with timestamps, and writes the time series to results.json
3. Execute smc_sim_data.py
   - Reads results.json and verifies reliability with sound and efficient SMC using fixed-k and sequential SMC and the Clopper-Pearson interval

### Design Time Digital Twins
The design time DTs are stored in dts/ . 
They can be executed on www.cpee.org as follows: 
1. Navigate to www.cpee.org
2. Start a demo
3. Create a new instance and press 'ok'
4. Load one of the CPEE process models in the respective design time DT below by "load testset"
5. Execute

#### Water kettle
The following processes start 1000x the water kettle design time DT for the respective initial parameter:
- 1000x 81.xml
- 1000x 90.xml
- 1000x 99.xml

Physical: Heating - Physical - Split.xml

Control: Heating - Control.xml

#### Water tank
The following processes start 1000x the water tank design time DT for the respective initial parameter:
- 1000x 11.xml
- 1000x 20.xml
- 1000x 29.xml

Physical: Waterlevel - Physical - Split.xml

Control: Waterlevel - Control.xml

#### Irrigation system
The following processes start 1000x the irrigation system design time DT:
- 1000x - Irrigation.xml

Physical: Soil Moisture - Physical.xml

Control: Soil Moisture - Control.xml

Environment: Soil Moisture - Environment *.xml


### Runtime Digital Twin
The runtime DT for the cotton candy system are available at https://zenodo.org/records/17226615. 
They could be executed on www.cpee.org similar to design time DTs, if the actual automated cotton candy machine would be ready for execution.
