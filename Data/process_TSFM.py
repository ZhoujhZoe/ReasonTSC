import json

# During dataset processing, TSFM converts class labels to zero-based indices. 
# Here, we revert them to their original UCR/UEA Archive labels.
mapping = {0: 1, 1: 10, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}

def process_txt_to_json(txt_filename, json_filename):
    data = []
    
    with open(txt_filename, 'r') as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            if line.startswith("Batch_idx") or not line.strip():
                continue
            
            parts = line.strip().split(', ', 3) 
            
            idx = int(parts[0])  # id
            predicted_label = int(parts[1])  # predicted
            real_label = int(parts[2])  # label
            logits_str = parts[3].strip('[]')  
            
            logits = [round(float(x), 2) for x in logits_str.split(", ")]
            
            mapped_predicted = mapping.get(predicted_label, predicted_label)
            mapped_label = mapping.get(real_label, real_label)

            mapped_logits = [0] * len(logits)
            for i, logit in enumerate(logits):
                mapped_idx = mapping.get(i, i)
                mapped_logits[mapped_idx - 1] = logit
            
            logits_str = f"[{', '.join(map(str, mapped_logits))}]"
            
            entry = {
                "id": idx, 
                "label": mapped_label,
                "predicted": mapped_predicted,
                "logits": logits_str  
            }
            data.append(entry)
    
    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main():
    txt_filename = './Chronos_MedicalImages.txt'
    json_filename = './Chronos_MedicalImages.json'
    process_txt_to_json(txt_filename, json_filename)

if __name__ == "__main__":
    main()