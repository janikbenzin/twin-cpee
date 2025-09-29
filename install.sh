#!/bin/sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install xes-yaml-pm4py-extension/pm4py-2.7.11.11-py3-none-any.whl
gem install cpee-logging-xes-yaml
cd sim_data/
curl --output runs_all_iot_systems.tgz "https://zenodo.org/api/records/16310813/files-archive"
tar -xvzf runs_all_iot_systems.tgz
cd ../run_data/
curl --output runs_cotton_candy.zip "https://zenodo.org/records/17226615/files/cotton-candy.zip?download=1"
unzip runs_cotton_candy.zip
