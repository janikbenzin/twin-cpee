import pm4py
import json
import traceback
import os

EXPERIMENTS = ["irrigation_batch_1", "heating_batch_81", "heating_batch_90", "heating_batch_99", "waterlevel_batch_11",
               "waterlevel_batch_20", "waterlevel_batch_29", "cotton_candy"]
TIME = "time"
TEMP = "temperature"
LEVEL = "level"
VAR = "variable"#
count = 0
variables_all = {exp: {} for exp in EXPERIMENTS}

### Real-world cotton candy runtime DT
batches = os.listdir("run_data/")
exp = EXPERIMENTS[-1]
for batch in batches:
    PATH_PREFIX = f"run_data/{batch}/"
    with open(PATH_PREFIX + 'index.txt') as f:
        indented_text = f.read()

    t = [line for line in indented_text.splitlines() if line.strip()]
    subprocesses = {ot.strip().split('(')[1].split(')')[0].strip(): ot.strip().split('(')[0].strip() for ot in
                    indented_text.splitlines() if ot.strip()}
    for i, (sid, sname) in enumerate(subprocesses.items()):
        filename = f"{PATH_PREFIX}{sid}.xes.yaml"
        if "Data Collection" in sname or "Cottonbot - Digital Twin" in sname:
            try:
                log = pm4py.read_yaml(filename)
                first_ts = log[((log["concept:name"] == "Create Cotton Candy")) & (log["cpee:lifecycle:transition"] == "activity/calling")]["time:timestamp"].to_list()[0]
                last_ts = log[((log["concept:name"] == "Create Cotton Candy")) & (log["cpee:lifecycle:transition"] == "activity/done")]["time:timestamp"].to_list()[-1]
                df = log[((log["concept:name"] == "Get the Environment Data")) & (log["cpee:lifecycle:transition"] == "dataelements/change")]
                times = list(df["time:timestamp"])
                less_than = [i for i,v in enumerate(times) if v < first_ts]
                less_i = max(less_than)
                greater_than = [i for i,v in enumerate(times) if v > last_ts]
                greater_i = min(greater_than)
                temperatures = list(df["data"].apply(lambda cell: cell["children"][0][1]["children"]["value"]["children"]["IrO"]))
                times = [str(j) for j in list(df["time:timestamp"])]
                variables_all[exp][sid] = {TIME: times[less_i+1:greater_i-1], VAR: temperatures[less_i +1:greater_i-1]}
            except Exception as e:
                print(f"Exception in {exp} {sid} {sname}: {str(e)}\n")
                traceback.print_exc()



for exp in EXPERIMENTS[:-1]:
    PATH_PREFIX = f"sim_data/{exp}/"
    with open(PATH_PREFIX + 'index.txt') as f:
        indented_text = f.read()

    t = [line for line in indented_text.splitlines() if line.strip()]
    subprocesses = {ot.strip().split('(')[1].split(')')[0].strip(): ot.strip().split('(')[0].strip() for ot in
                    indented_text.splitlines() if ot.strip()}
    for i, (sid, sname) in enumerate(subprocesses.items()):
        filename = f"{PATH_PREFIX}{sid}.xes.yaml"
        if "1000" not in sname:
            try:
                log = pm4py.read_yaml(filename)
                if exp.startswith("heating") and sname.startswith("Heating - Physical"):
                    df = log[((log["concept:name"] == "Heat") | (log["concept:name"] == "Cool")) & (
                                log["cpee:lifecycle:transition"] == "dataelements/change")]
                    times = [str(j) for j in list(df["time:timestamp"])]
                    temperatures = list(df["data"].apply(lambda cell: cell["children"][0][1]["children"]["value"]))
                    variables_all[exp][sid] = {TIME: times, VAR: temperatures}
                elif exp.startswith("waterlevel") and sname.startswith("Waterlevel - Physical"):
                    df = log[((log["concept:name"] == "On") | (log["concept:name"] == "Off")) & (
                            log["cpee:lifecycle:transition"] == "dataelements/change")]
                    times = [str(j) for j in list(df["time:timestamp"])]
                    levels = list(df["data"].apply(lambda cell: cell["children"][0][1]["children"]["value"]))
                    variables_all[exp][sid] = {TIME: times, VAR: levels}
                elif exp.startswith("irrigation") and sname.startswith("Soil Moisture - Physical"):
                    df = log[((log["concept:name"] == "Dry") | (log["concept:name"] == "Water")) & (
                            log["cpee:lifecycle:transition"] == "dataelements/change")]
                    times = [str(j) for j in list(df["time:timestamp"])]
                    levels = list(df["data"].apply(lambda cell: cell["children"][0][1]["children"]["value"]))
                    if len(levels) == 0:
                        count += 1
                        print(sid,sname)
                    variables_all[exp][sid] = {TIME: times, VAR: levels}
            except Exception as e:
                print(f"Exception in {exp} {sid} {sname}: {str(e)}\n")
                traceback.print_exc()


with open("results/results.json", "w") as f:
    json.dump(variables_all, f)