import json
import os

# Find the most recent minions log file
log_dir = "minions_logs"
log_files = [f for f in os.listdir(log_dir) if f.endswith("_minions.json")]
if not log_files:
    print("No minions log files found")
    exit(1)

# Sort by modification time and get the most recent
log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
latest_log = os.path.join(log_dir, log_files[0])

print(f"Analyzing log file: {latest_log}")
print("="*80)

# Load the log file
with open(latest_log, 'r', encoding='utf-8') as f:
    log_data = json.load(f)

# Look for worker outputs
for entry in log_data['conversation']:
    if entry['user'] == 'worker' and entry.get('output'):
        print("\nWorker outputs found:")
        outputs = entry['output']
        
        # Show first few outputs
        for i, output in enumerate(outputs[:3]):
            print(f"\nWorker output {i+1}:")
            print("-"*40)
            if len(output) > 500:
                print(output[:500] + "...")
            else:
                print(output)
            print("-"*40)
            
        print(f"\nTotal worker outputs: {len(outputs)}")
        
        # Check if any outputs are valid JSON
        valid_json_count = 0
        for output in outputs:
            try:
                json.loads(output)
                valid_json_count += 1
            except:
                pass
        
        print(f"Valid JSON outputs: {valid_json_count}/{len(outputs)}") 