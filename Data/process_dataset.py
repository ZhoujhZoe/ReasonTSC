import json

def process_ts_to_json(ts_filename, json_filename):
    data = []
    
    with open(ts_filename, 'r') as ts_file:
        lines = ts_file.readlines()
        
        for idx, line in enumerate(lines):
            line = line.strip()
            
            sample_data, label = line.rsplit(":", 1)
            
            
            sample = ','.join([f"{float(num):.3f}" for num in sample_data.split(',')])
            
            entry = {
                "id": idx,
                "label": int(label),
                "sample": sample
            }
            
            data.append(entry)
    
    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


ts_filename = './MiddlePhalanxOutlineAgeGroup_TRAIN.ts'
json_filename = './MiddlePhalanxOutlineAgeGroup_TRAIN.json'
process_ts_to_json(ts_filename, json_filename)