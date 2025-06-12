import pandas as pd

# Load the cleaned data
df = pd.read_csv('security-test/test_data_cleaned.csv')

# Check distribution in first few batches
batch_size = 500
num_batches_to_check = 5

print("Batch Distribution Analysis")
print("="*50)

for i in range(num_batches_to_check):
    start = i * batch_size
    end = min(start + batch_size, len(df))
    batch = df.iloc[start:end]
    
    malicious = (batch['label_binary'] == True).sum()
    benign = (batch['label_binary'] == False).sum()
    
    print(f"\nBatch {i+1} (rows {start+1}-{end}):")
    print(f"  Malicious: {malicious} ({malicious/len(batch)*100:.1f}%)")
    print(f"  Benign: {benign} ({benign/len(batch)*100:.1f}%)")

# Overall distribution
print(f"\n\nOverall Distribution:")
print(f"Total samples: {len(df)}")
print(f"Malicious: {(df['label_binary'] == True).sum()} ({(df['label_binary'] == True).sum()/len(df)*100:.1f}%)")
print(f"Benign: {(df['label_binary'] == False).sum()} ({(df['label_binary'] == False).sum()/len(df)*100:.1f}%)") 