import pandas as pd
from rdflib import Graph
import re
from glob import glob
import datetime
import numpy as np
from typing import Tuple
from tqdm import tqdm

##################

def fix_time(x):
    try:
         return datetime.datetime.strptime(x, '%H:%M:%S.%f').time()
    except:
        return datetime.datetime.strptime(x, '%H:%M:%S').time()

def combine(x):
    try:
        d = x.astype(float)
        if len(d) > 0:
            return d.mean(), d.min(), d.max()
        else:
            return np.nan
    except:
        return list(x.values)

def combine_pred(x):
    try:
        d = x.astype(float)
        if len(d) > 0:
            return d.mean(), d.min(), d.max()
        else:
            return np.nan
    except:
        return set(x.values)

uuid_room_map = {}
def generate_room_uuid(room, part):
    if part not in uuid_room_map:
        uuid_room_map[part] = {}
    if room not in uuid_room_map[part]:
        uuid_room_map[part][room] = 0
    result = uuid_room_map[part][room]
    uuid_room_map[part][room] += 1
    return 'https://dahcc.idlab.ugent.be/Protego/'+part+'/'+room+'/state'+str(result)

uuid_wearable_map = {}
def generate_wearable_uuid(wearable, part):
    if part not in uuid_wearable_map:
        uuid_wearable_map[part] = {}
    if wearable not in uuid_wearable_map[part]:
        uuid_wearable_map[part][wearable] = 0
    result = uuid_wearable_map[part][wearable]
    uuid_wearable_map[part][wearable] += 1
    return 'https://dahcc.idlab.ugent.be/Protego/'+part+'/'+wearable+'/state'+str(result)

uuid_event_map = {}
def generate_event_uuid(part :str)-> Tuple[str, str]:
    if part not in uuid_event_map:
        uuid_event_map[part] = 0
    result = uuid_event_map[part]
    uuid_event_map[part] += 1
    return 'https://dahcc.idlab.ugent.be/Protego/'+part+'/state'+str(uuid_event_map[part]), 'https://dahcc.idlab.ugent.be/Protego/'+part+'/state'+str(result)

uuid_obs_map = {}
def generate_obs_uuid(part:str) -> str:
    if part not in uuid_obs_map:
        uuid_obs_map[part] = 0
    result = uuid_obs_map[part]
    uuid_obs_map[part] += 1
    return 'https://dahcc.idlab.ugent.be/Protego/'+part+'/state'+str(result)

aqura_conv = {'Badkamer zorgkamer':  'careroom',
'Zorgkamer': 'careroom',
'Hall beneden schuifdeur': 'mainhall',
'Eethoek': 'living',
'Keuken':'kitchen',
'Zithoek': 'living',
'Traphal': 'hallway1',
 'Gang vergaderzalen': 'hallway1',
'Gang overloop' :  'hallway1',
'Slaapkamer ouders'  : 'masterbedroom',
'Badkamer boven' : 'bathroom',
'Toilet beneden' : 'toilet1',
 'Sas kelder' : 'basement',
'Lavabo toilet boven': 'toilet2',
'Toilet boven': 'toilet2',
'Zorgkamer massif':'office',
'Garage' : 'garage',
'Berging' : 'laundryroom',
'Kleine vergaderzaal' : 'hallway1'
}
METRIC_TYPES_CONTEXT = {
    "airquality.co2": float,
    "airquality.voc_total": float,
    "energy.consumption": float,
    "energy.power": float,
    "environment.blind": float,
    "environment.button": bool,
    "environment.dimmer": float,
    "environment.light": float,
    "environment.lightswitch": bool,
    "environment.motion": bool,
    "environment.open": bool,
    "environment.relativehumidity": float,
    "environment.relay": bool,
    "environment.temperature": float,
    "environment.voltage": float,
    "environment.waterRunning::bool": bool,
    "mqtt.lastMessage": str,
    "org.dyamand.aqura.AquraLocationState_Protego_User": "ext",
    "org.dyamand.types.airquality.CO2": float,
    "org.dyamand.types.common.AtmosphericPressure": float,
    "org.dyamand.types.common.Loudness": float,
    "org.dyamand.types.common.RelativeHumidity": float,
    "org.dyamand.types.common.Temperature": float,
    "people.presence.detected": bool,
    "people.presence.numberDetected": float,
    "weather.pressure": float,
    "weather.rainrate": float,
    "weather.windspeed": float
}

METRIC_TYPES_WEARABLE = {
    "org.dyamand.types.health.BodyTemperature": float,
    "org.dyamand.types.common.Load": float,
    "org.dyamand.types.health.DiastolicBloodPressure": float,
    "org.dyamand.types.health.HeartRate": float,
    "org.dyamand.types.health.SystolicBloodPressure": float,
    "smartphone.acceleration.x": float,
    "smartphone.acceleration.y": float,
    "smartphone.acceleration.z": float,
    "smartphone.ambient_light": float,
    "smartphone.ambient_noise.amplitude": float,
    "smartphone.ambient_noise.frequency": float,
    "smartphone.application": str,
    "smartphone.gravity.x": float,
    "smartphone.gravity.y": float,
    "smartphone.gravity.z": float,
    "smartphone.gyroscope.x": float,
    "smartphone.gyroscope.y": float,
    "smartphone.gyroscope.z": float,
    "smartphone.keyboard": str,
    "smartphone.linear_acceleration.x": float,
    "smartphone.linear_acceleration.y": float,
    "smartphone.linear_acceleration.z": float,
    "smartphone.location.accuracy": float,
    "smartphone.location.altitude": float,
    "smartphone.location.bearing": float,
    "smartphone.location.latitude": float,
    "smartphone.location.longitude": float,
    "smartphone.magnetometer.x": float,
    "smartphone.magnetometer.y": float,
    "smartphone.magnetometer.z": float,
    "smartphone.proximity": float,
    "smartphone.rotation.x": float,
    "smartphone.rotation.y": float,
    "smartphone.rotation.z": float,
    "smartphone.screen": str,
    "smartphone.sleepAPI.API_confidence": float,
    "smartphone.sleepAPI.model_type": str,
    "smartphone.sleepAPI.t_start": float,
    "smartphone.sleepAPI.t_stop": float,
    "smartphone.step": float,
    "wearable.acceleration.x": float,
    "wearable.acceleration.y": float,
    "wearable.acceleration.z": float,
    "wearable.battery_level": float,
    "wearable.bvp": float,
    "wearable.gsr": float,
    "wearable.ibi": float,
    "wearable.on_wrist": bool,
    "wearable.skt": float
}

TYPE_MAP = {
    float: "<http://www.w3.org/2001/XMLSchema#float>",
    int: "<http://www.w3.org/2001/XMLSchema#integer>",
    bool: "<http://www.w3.org/2001/XMLSchema#boolean>",
    str: "<http://www.w3.org/2001/XMLSchema#string>"
}
VALUE_TEMPLATE = "\"%s\"^^%s"

def map_value(value, metric_id):
    value_type, metric_group = float, None
    if metric_id in METRIC_TYPES_CONTEXT:
        value_type, metric_group = METRIC_TYPES_CONTEXT[metric_id], "CONTEXT"
    elif metric_id in METRIC_TYPES_WEARABLE:
        value_type, metric_group = METRIC_TYPES_WEARABLE[metric_id], "WEARABLE"

    if value_type == bool:
        processed_value = "true" if int(value) == 1 else "false"
    else:
        processed_value = value
    if value_type == "ext":

        if value in aqura_conv:
            processed_value = aqura_conv[value]
            value_type=str
        else:
            print(value)
    return VALUE_TEMPLATE % (processed_value, TYPE_MAP[value_type]), metric_group


def create_event(ff, part, time, begin_event, preds):
    ff = ff[ff.Sensor != "velbus.B8.EnergyMeter3"]
    ff = ff.replace({'Metric':{'org.dyamand.aqura.AquraLocationState_Protego User':'org.dyamand.aqura.AquraLocationState_Protego_User'}})
    activities = set()
    for sublist in ff.label.values:
        for item in sublist:
            activities.update(set(item.split(', ')))

    routines = set()
    for value in ff.routine.values:
        if len(value)>0:
            routines.add(list(value)[0])

    with open('event_graph_30_seconds.nt', 'a') as f:
        event, prev = generate_event_uuid(part)
        if prev:
            f.write('<%s> <http://example.org/hasPrevious> <%s> .\n'%(event, prev))
        if begin_event:
            f.write('<%s> <http://example.org/isBeginEvent> "True"^^<http://www.w3.org/2001/XMLSchema#boolean> .\n'%(event))
        f.write('<%s> <https://saref.etsi.org/core/hasTimestamp> "%s"^^<http://www.w3.org/2001/XMLSchema#dateTime> .\n'%(event, time))

        for a in activities:
            f.write('<%s> <https://saref.etsi.org/core/hasActivity> "%s" .\n' % (event, a))

        for r in routines:
            f.write('<%s> <http://example.org/hasRoutine> "%s" .\n' % (event, r))


        if len(preds)>0:
            for p in preds['ground_truth'].values[0]:
                f.write('<%s> <http://example.org/hasSmartphoneActivityPrediction> "%s" .\n'%(event, p))

        for r in rooms_types:
            room_state = generate_room_uuid(r.split('/')[-1],part)
            f.write("<%s> <%s> <%s> .\n"%(event, "https://dahcc.idlab.ugent.be/Protego/"+rooms_types[r], room_state))

            if r in result_appliances:
                for ap in result_appliances[r]:
                    res_frame = ff.loc[(ff.Sensor==result_appliances[r][ap] )& (ff.Metric.str.contains("energy.power|environment"))]
                    if len(res_frame)>0:
                        for res in res_frame[["Value","Sensor"]].values:
                            d = list(res)[0]
                            if d[0] > 0:
                                obs = generate_obs_uuid(part)
                                f.write("<%s> <%s> <%s> .\n"%(room_state, "https://dahcc.idlab.ugent.be/Protego/"+ap, obs))
                                f.write('<%s> <%s> "%s" .\n'%(obs, "https://saref.etsi.org/core/hasValue", "on"))
                                f.write('<%s> <%s> <%s> .\n'%(obs, "https://saref.etsi.org/core/measurementMadeBy", "https://dahcc.idlab.ugent.be/Homelab/SensorsAndActuators/"+list(res)[1]))
                            else:
                                obs = generate_obs_uuid(part)
                                f.write("<%s> <%s> <%s> .\n"%(room_state, "https://dahcc.idlab.ugent.be/Protego/"+ap, obs))
                                f.write('<%s> <%s> "%s" .\n'%(obs, "https://saref.etsi.org/core/hasValue", "off"))
                                f.write('<%s> <%s> <%s> .\n'%(obs, "https://saref.etsi.org/core/measurementMadeBy", "https://dahcc.idlab.ugent.be/Homelab/SensorsAndActuators/"+list(res)[1]))
            if r in result_room_states:
                for o in result_room_states[r]:
                    res_frame = ff.loc[(ff.Sensor==result_room_states[r][o][0])&(ff.Metric==result_room_states[r][o][1])]
                    for res in res_frame[["Value","Sensor"]].values:
                        d = list(res)[0]
                        obs = generate_obs_uuid(part)
                        f.write("<%s> <%s> <%s> .\n"%(room_state, "https://dahcc.idlab.ugent.be/Protego/"+o, obs))
                        f.write('<%s> <%s> "%s"^^<http://www.w3.org/2001/XMLSchema#float> .\n'%(obs, "https://dahcc.idlab.ugent.be/Protego/hasMeanValue", d[0]))
                        f.write('<%s> <%s> "%s"^^<http://www.w3.org/2001/XMLSchema#float> .\n'%(obs, "https://dahcc.idlab.ugent.be/Protego/hasMinValue", d[1]))
                        f.write('<%s> <%s> "%s"^^<http://www.w3.org/2001/XMLSchema#float> .\n'%(obs, "https://dahcc.idlab.ugent.be/Protego/hasMaxValue", d[2]))
                        f.write('<%s> <%s> <%s> .\n'%(obs, "https://saref.etsi.org/core/measurementMadeBy", "https://dahcc.idlab.ugent.be/Homelab/SensorsAndActuators/"+list(res)[1]))

        for w in wearable_states:
            wearable_state = generate_room_uuid(w,part)
            f.write("<%s> <%s> <%s> .\n"%(event, "https://dahcc.idlab.ugent.be/Protego/Person"+w, wearable_state))
            for device in wearable_states[w]:
                for metric in wearable_states[w][device]:
                    data = wearable_states[w][device][metric]
                    res_frame = ff.loc[(ff.Sensor==data[0])&(ff.Metric==data[1])]
                    if len(res_frame)>0:
                        for res in res_frame[["Value","Sensor"]].values:
                            if isinstance(res[0], list):
                                for d in res[0]:
                                    obs = generate_obs_uuid(part)
                                    o = 'has'+metric.replace('_',' ').replace('.',' ').title().replace(' ','')+"Observation"
                                    f.write("<%s> <%s> <%s> .\n"%(wearable_state, "https://dahcc.idlab.ugent.be/Protego/"+o, obs))
                                    f.write('<%s> <%s> %s .\n'%(obs, "https://dahcc.idlab.ugent.be/Protego/hasMeanValue", map_value(d, metric)[0]))
                                    f.write('<%s> <%s> <%s> .\n'%(obs, "https://saref.etsi.org/core/measurementMadeBy", "https://dahcc.idlab.ugent.be/Homelab/SensorsAndActuators/"+res[1]))
                            else:
                                obs = generate_obs_uuid(part)
                                o = 'has'+metric.replace('_',' ').replace('.',' ').title().replace(' ','')+"Observation"
                                d=res[0]
                                f.write("<%s> <%s> <%s> .\n"%(wearable_state, "https://dahcc.idlab.ugent.be/Protego/"+o, obs))
                                f.write('<%s> <%s> "%s"^^<http://www.w3.org/2001/XMLSchema#float> .\n'%(obs, "https://dahcc.idlab.ugent.be/Protego/hasMeanValue", d[0]))
                                f.write('<%s> <%s> "%s"^^<http://www.w3.org/2001/XMLSchema#float> .\n'%(obs, "https://dahcc.idlab.ugent.be/Protego/hasMinValue", d[1]))
                                f.write('<%s> <%s> "%s"^^<http://www.w3.org/2001/XMLSchema#float> .\n'%(obs, "https://dahcc.idlab.ugent.be/Protego/hasMaxValue", d[2]))
                                f.write('<%s> <%s> <%s> .\n'%(obs, "https://saref.etsi.org/core/measurementMadeBy", "https://dahcc.idlab.ugent.be/Homelab/SensorsAndActuators/"+list(res)[1]))
##################

g = Graph()
g.parse("/Users/bramsteenwinckel/Documents/repos/DAHCC-Ontology/instantiated_examples/_Homelab.owl")

query = """
SELECT ?room ?room_type ?floor
WHERE {
    ?floor <https://saref.etsi.org/saref4bldg/isSpaceOf> <https://dahcc.idlab.ugent.be/Homelab/SensorsAndActuators/homelab> .
    ?room <https://saref.etsi.org/saref4bldg/isSpaceOf> ?floor .
    ?room a ?room_type .
    Filter(?room_type != <http://www.w3.org/2002/07/owl#NamedIndividual>) .
    Filter(?room_type != <https://dahcc.idlab.ugent.be/Ontology/SensorsAndActuators/MultipurposeRoom>) .
    Filter(?room_type != <https://dahcc.idlab.ugent.be/Ontology/SensorsAndActuators/Garage>) .
    Filter(?room != <https://dahcc.idlab.ugent.be/Homelab/SensorsAndActuators/smallbedroom>) .
}"""

qres = g.query(query)
rooms_types = {}
for row in qres:
    rooms_types[row.room.toPython()] = row.floor.split('/')[-1]+''+row.room_type.split('/')[-1]

appliances = {}
for room in rooms_types:
    query = """
    SELECT ?appliance ?appliance_type ?sensor
    WHERE {
        ?appliance <https://saref.etsi.org/saref4bldg/isContainedIn> <%s> .
        ?sensor <https://dahcc.idlab.ugent.be/Ontology/Sensors/analyseStateOf> ?appliance .
        ?appliance a ?appliance_type .
        Filter(?appliance_type != <http://www.w3.org/2002/07/owl#NamedIndividual>) .
    }"""%(room)

    qres = g.query(query)
    for row in qres:
        if room not in appliances:
            appliances[room] = {}
        tp = row.appliance_type.split('/')[-1].split('#')[-1]
        if tp not in appliances[room]:
            appliances[room][tp] = set()
        appliances[room][tp].add((row.sensor.split('/')[-1],row.appliance.toPython()))

result_appliances = {}
for room in appliances:
    result_appliances[room] = {}
    for a in appliances[room]:
        if len(appliances[room][a])==1:
            result_appliances[room]['has'+a+"State"] = list(appliances[room][a])[0][0]
        else:
            sensors = [re.sub('([a-zA-Z])', lambda x: x.groups()[0].upper(), x[1].split('/')[-1].replace('_',''), 1) for x in appliances[room][a]]
            for s in sensors:
                result_appliances[room]['has'+a+"From"+s+"State"] = list(appliances[room][a])[0][0]


room_states = {}
for room in rooms_types:
    query = """
    SELECT ?sensor ?sensor_type ?property
    WHERE {
        ?sensor <https://saref.etsi.org/saref4bldg/isContainedIn> <%s> .
        ?sensor <https://dahcc.idlab.ugent.be/Ontology/Sensors/analyseStateOf> <%s> .
        ?sensor <https://saref.etsi.org/core/measuresProperty> ?property .
        ?sensor a ?sensor_type .
        Filter(?sensor_type != <http://www.w3.org/2002/07/owl#NamedIndividual>) .
    }"""%(room,room)

    qres = g.query(query)
    for row in qres:
        if room not in room_states:
            room_states[room] = {}
        p = row.property.split('/')[-1].split('#')[-1]
        if p not in room_states[room]:
            room_states[room][p] = set()
        room_states[room][p].add(row.sensor.split('/')[-1])

result_room_states = {}
for room in room_states:
    result_room_states[room] = {}
    for a in room_states[room]:
        if len(room_states[room][a]) == 1:
            result_room_states[room]['has'+a.replace('.',' ').title().replace(' ','')+"Observation"] = (list(room_states[room][a])[0],a)
        else:
            for i in range(len(room_states[room][a])):
                result_room_states[room]['has'+a.replace('.',' ').title().replace(' ','')+"ObservationNr"+str(i)] = (list(room_states[room][a])[i],a)


g = Graph()
g.parse("/Users/bramsteenwinckel/Documents/repos/DAHCC-Ontology/instantiated_examples/_Homelab.owl")
g.parse("/Users/bramsteenwinckel/Documents/repos/DAHCC-Ontology/instantiated_examples/_HomelabWearable.owl")
query = """
SELECT ?device ?device_type ?property ?sensor
WHERE {
    ?device <https://saref.etsi.org/core/consistsOf> ?sensor.
    ?device a ?device_type .
    ?sensor <https://saref.etsi.org/core/measuresProperty> ?property .
    Filter(?device_type != <http://www.w3.org/2002/07/owl#NamedIndividual>) .
}"""

wearable_states = {}
qres = g.query(query)
for row in qres:
    device = row.device.toPython()
    device_type = row.device_type.split('/')[-1]
    if device_type not in wearable_states:
        wearable_states[device_type] = {}
    if device not in wearable_states[device_type]:
        wearable_states[device_type][device] = {}
    p = row.property.split('/')[-1].split('#')[-1]
    if p not in wearable_states[device_type][device]:
        if device_type == 'Localisation':
            wearable_states[device_type][device][p] = (row.sensor.split('/')[-1],p,row.sensor.split('/')[-1])
        else:
            wearable_states[device_type][device][p] = (device.split('/')[-1],p,row.sensor.split('/')[-1])


#### main loop

li = []
for filename in glob('predictions/*/predictions.csv'):
    df = pd.read_csv(filename, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.timestamps = pd.to_datetime(frame.timestamps).dt.tz_localize(None)
frame.index = frame.timestamps
pred_a = frame.groupby(['participant']).resample("30s")['ground_truth'].apply(combine_pred).ffill().reset_index()

for user in tqdm(glob('/Users/bramsteenwinckel/Datasets/Protego_anom/*/')):
    df = pd.read_feather(user+'data.feather')
    lf = pd.read_csv(user+'/labels.csv')

    df.Timestamp = [datetime.datetime.combine(datetime.date(2000, 1, 1), x) for x in df.Timestamp]
    lf.start_time = lf.start_time.apply(fix_time)
    lf.end_time = lf.end_time.apply(fix_time)
    lf.start_time = [datetime.datetime.combine(datetime.date(2000, 1, 1), x) for x in lf.start_time]
    lf.end_time = [datetime.datetime.combine(datetime.date(2000, 1, 1), x) for x in lf.end_time]

    df.index = df.Timestamp

    gr_df = df.groupby(['Sensor', 'Metric']).resample("30s")['Value'].apply(combine).ffill().reset_index()
    gr_df = gr_df[gr_df["Metric"].str.contains("mqtt") == False]

    gr_df = gr_df.set_index('Timestamp').sort_index()
    gr_df['label'] = [set() for i in range(len(gr_df))]
    gr_df['routine'] = [set() for i in range(len(gr_df))]
    for i, row in lf.iterrows():
        gr_df.loc[str(row.start_time):str(row.end_time), 'label'].apply(lambda x: x.add(row.ontology_label))
        gr_df.loc[str(row.start_time):str(row.end_time), 'routine'].apply(lambda x: x.add(row.ontology_routine))

    gr_df.label = gr_df.label.apply(lambda x: x if len(x) > 0 else ['Unknown'])
    gr_df['ontology_routine'] = gr_df.routine.apply(lambda x: x if len(x) > 0 else ['Unknown'])
    gr_df = gr_df.dropna()

    gr_df = gr_df.sort_index()
    gr_df = gr_df.reset_index()

    gr_df['t_label'] = gr_df['label'].shift(-1)
    gr_df['p_label'] = gr_df['label'].shift(1)

    part = user.split('/')[-2]

    begin_events = gr_df[(gr_df['label'] != gr_df['p_label']) & (gr_df['label'] != 'Unknown')]['Timestamp'].values
    for i, event in gr_df.groupby('Timestamp'):
        preds = pred_a[(pred_a.timestamps == i) & (pred_a.participant == ''.join(part.split('_')))]
        create_event(event, part, i, i in begin_events, preds)



