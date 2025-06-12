# Security Test: Minions vs GPT-4o for Network Packet Classification

This directory contains tests comparing the Minions protocol against GPT-4o for classifying network packets as benign or malicious.

## Overview

The test evaluates both systems on a dataset of ~95,000 network packets (roughly 50/50 benign/malicious split) to determine:
- Classification accuracy (precision, recall, F1 score)
- Processing speed
- Cost efficiency

## Dataset

The test uses network packet data with the following characteristics:
- **Total samples**: 94,901
- **Malicious packets**: 46,995 (various attack types)
- **Benign packets**: 47,906
- **Features**: Connection states, IPs, ports, protocols, byte counts, etc.

Attack types include:
- Credential Access (T1110)
- Reconnaissance (T1595)
- Defense Evasion (T1078)
- Initial Access (T1190)
- Exfiltration (T1048)

## Setup

1. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn python-dotenv
   ```

2. **Set up API keys**:
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

3. **Set up Ollama** (optional but recommended):
   ```bash
   # Install Ollama from https://ollama.com/download
   # Start Ollama server
   ollama serve
   
   # Pull the llama3.2 model
   ollama pull llama3.2
   ```

## Running the Tests

### 1. Test Setup
First, verify your setup is working:
```bash
python security-test/test_setup.py
```

### 2. Run Comparison
Run the full comparison:
```bash
python security-test/test_security_comparison.py
```

Options:
- `--max-batches N`: Limit to N batches (for testing)
- `--no-ollama`: Use GPT-4o-mini instead of Ollama for local model

Example for testing with 2 batches:
```bash
python security-test/test_security_comparison.py --max-batches 2
```

## How It Works

1. **Data Processing**: Packets are processed in batches of 500
2. **Prompt**: Both models receive the same security expert prompt
3. **Classification**: Models return JSON with malicious packet numbers
4. **Evaluation**: Results are compared against ground truth labels

### Minions Protocol
- **Local Model**: Processes packets initially (Ollama/llama3.2 or GPT-4o-mini)
- **Remote Model**: GPT-4o supervises and refines results
- **Communication**: Up to 2 rounds of local-remote interaction

### GPT-4o Direct
- Single API call to GPT-4o with all packet data

## Output

The test produces:
1. **Console output**: Real-time progress and final metrics
2. **JSON results**: Detailed results saved to `results_TIMESTAMP.json`

### Metrics Reported
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix (TP, TN, FP, FN)
- Processing time per batch
- Comparative analysis

## Expected Results

Based on the architecture:
- **Accuracy**: Both should achieve high accuracy (>90%)
- **Speed**: Minions may be slower due to multiple rounds
- **Cost**: Minions should be 50-70% cheaper due to local processing

## Troubleshooting

1. **Ollama not found**: 
   - Ensure Ollama is running: `ollama serve`
   - Falls back to GPT-4o-mini automatically

2. **API key errors**:
   - Check `.env` file exists and contains valid keys
   - Verify key permissions for GPT-4o access

3. **Memory issues**:
   - Use `--max-batches` to process fewer samples
   - Reduce batch size in the code if needed

## Files

- `test_security_comparison.py`: Main comparison script
- `test_setup.py`: Setup verification script
- `test_data_cleaned.csv`: Cleaned dataset (94,901 samples)
- `results_*.json`: Output files with detailed results 