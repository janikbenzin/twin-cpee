#!/bin/sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install xes-yaml-pm4py-extension/pm4py-2.7.11.11-py3-none-any.whl
gem install cpee-logging-xes-yaml
# Add curl from zenodo
cd sim_data/
tar -xvzf runs_all_iot_systems.tgz

