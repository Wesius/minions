import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dotenv import load_dotenv

# Add parent directory to path to import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients.openai import OpenAIClient
from minions.clients.ollama import OllamaClient
from minions.clients.distributed_inference import DistributedInferenceClient
from minions.minions import Minions

# Load environment variables
load_dotenv()

class SecurityTestComparison:
    def __init__(self, data_path: str = "security-test/test_data_cleaned.csv", provider: str = "ollama"):
        """Initialize the security test comparison framework.
        
        Args:
            data_path: Path to the test data CSV file
            provider: Local model provider - "ollama", "distributed", or "openai" (uses gpt-4o-mini)
        """
        self.data_path = data_path
        self.batch_size = 250
        self.provider = provider
        self.results = {
            'minions': {'predictions': [], 'times': []},
            'gpt4o': {'predictions': [], 'times': []}
        }
        
        # Token tracking for cost calculation
        self.token_usage = {
            'gpt4o': {'input_tokens': 0, 'output_tokens': 0},
            'minions': {
                'remote': {'input_tokens': 0, 'output_tokens': 0},
                'local': {'input_tokens': 0, 'output_tokens': 0}
            }
        }
        
        # OpenAI pricing (as of 2024)
        self.pricing = {
            'gpt-4o': {
                'input': 2.50 / 1_000_000,  # $2.50 per 1M input tokens
                'output': 10.00 / 1_000_000  # $10.00 per 1M output tokens
            },
            'gpt-4o-mini': {
                'input': 0.15 / 1_000_000,  # $0.15 per 1M input tokens
                'output': 0.60 / 1_000_000   # $0.60 per 1M output tokens
            }
        }
        
        # Load and prepare data
        print("Loading test data...")
        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df)} samples")
        
        # Shuffle the data to ensure proper mix of malicious/benign in each batch
        print("Shuffling data for balanced batches...")
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Features to exclude (labels and obvious indicators)
        self.exclude_columns = [
            'label_binary', 'label_tactic', 'label_technique', 'label_cve',
            'datetime', 'uid'  # These might give away temporal patterns
        ]
        
        # Prepare feature columns
        self.feature_columns = [col for col in self.df.columns if col not in self.exclude_columns]
        
        # Initialize clients
        print("Initializing clients...")
        self.gpt4o_client = OpenAIClient(model_name="gpt-4o")
        
        # For Minions, try to use Ollama for local model
        if provider == "ollama":
            try:
                print("Attempting to connect to Ollama...")
                # Use async mode for better parallel processing with Minions
                local_client = OllamaClient(
                    model_name="llama3.2",
                    use_async=True,  # Enable async for parallel worker calls
                    temperature=0.0,
                    max_tokens=2048,
                    num_ctx=8192  # Reduce context size to avoid memory issues
                )
                print("Successfully connected to Ollama with llama3.2")
            except Exception as e:
                print(f"Failed to connect to Ollama: {e}")
                print("Falling back to GPT-4o-mini as local model")
                local_client = OpenAIClient(model_name="gpt-4o-mini")
        elif provider == "distributed":
            print("Using Distributed Inference for local model")
            try:
                # Get configuration from environment or use defaults
                coordinator_url = os.getenv("MINIONS_COORDINATOR_URL", "http://localhost:8080")
                api_key = os.getenv("MINIONS_API_KEY")
                model_name = "llama3.2:1b" # Optional specific model
                
                local_client = DistributedInferenceClient(
                    model_name=model_name,  # None means auto-select
                    temperature=0.0,
                    max_tokens=2048,
                    base_url=coordinator_url
                )
                print(f"Connected to Distributed Inference at {coordinator_url}")
                if model_name:
                    print(f"Requesting model: {model_name}")
                else:
                    print("Model will be auto-selected by coordinator")
            except Exception as e:
                print(f"Failed to connect to Distributed Inference: {e}")
                print("Falling back to GPT-4o-mini as local model")
                local_client = OpenAIClient(model_name="gpt-4o-mini")
        else:
            print("Using GPT-4o-mini as local model")
            local_client = OpenAIClient(model_name="gpt-4o-mini")
        
        # Initialize Minions with custom worker prompt for JSON output
        worker_prompt = """Here is a document excerpt:

{context}

--------------------------------
And here is your task:

{task}

--------------------------------
And here is additional higher-level advice on how to approach the task:

{advice}

--------------------------------

IMPORTANT: You MUST respond with a valid JSON object in exactly this format:
{{
    "explanation": "Your reasoning here",
    "citation": null,
    "answer": "Your answer here"
}}

Do not include any text before or after the JSON. Only output the JSON object.
Your response:"""
        
        self.minions = Minions(
            local_client=local_client,
            remote_client=self.gpt4o_client,
            max_rounds=5,
            worker_prompt_template=worker_prompt,
            log_dir="security-test/minions_logs"
        )
        
    def prepare_packet_batch(self, start_idx: int, end_idx: int) -> Tuple[str, List[bool]]:
        """Prepare a batch of packets for classification."""
        batch_df = self.df.iloc[start_idx:end_idx]
        
        # Create packet descriptions
        packets = []
        true_labels = []
        
        for idx, row in batch_df.iterrows():
            # Create a structured description of the packet
            packet_desc = f"Packet {idx - start_idx + 1}: "
            packet_features = []
            
            for col in self.feature_columns:
                value = row[col]
                if pd.notna(value):  # Only include non-null values
                    packet_features.append(f"{col}={value}")
            
            packet_desc += ", ".join(packet_features)
            packets.append(packet_desc)
            true_labels.append(row['label_binary'])
        
        # Combine into a single context string
        context = "\n".join(packets)
        
        return context, true_labels
    
    def create_classification_prompt(self) -> str:
        """Create the prompt for packet classification."""
        return """You are a network security expert analyzing network packets for potential malicious activity.

You will be given a list of network packets, each with a numerical label and various features.
Your task is to identify which packets are malicious based on their characteristics.

Common indicators of malicious activity include:
- Unusual port numbers or protocols
- Suspicious connection patterns
- Abnormal byte counts or packet sizes
- Connections to/from unusual IPs
- Patterns consistent with scanning, brute force, or data exfiltration

IMPORTANT: You must respond with a valid JSON object in the following format:
{
    "explanation": "Brief explanation of your analysis approach",
    "citation": null,
    "answer": "[1, 5, 12, ...]"
}

Where "answer" contains a JSON array of the numerical labels of packets you believe are malicious.
If no packets appear malicious, use an empty array: []

Remember: You MUST return valid JSON with these exact fields."""
    
    def parse_model_response(self, response: str, batch_size: int) -> List[bool]:
        """Parse model response to extract malicious packet predictions."""
        predictions = [False] * batch_size
        
        try:
            # Try to parse as JSON
            data = json.loads(response)
            
            # Handle JobOutput format (from Minions)
            if isinstance(data, dict) and 'answer' in data:
                answer = data['answer']
                if answer is None or answer == "null" or answer == "":
                    return predictions
                
                # Parse the answer field which should contain a JSON array string
                if isinstance(answer, str):
                    try:
                        malicious_list = json.loads(answer)
                    except:
                        # If it's not valid JSON, try to extract numbers
                        import re
                        numbers = re.findall(r'\d+', answer)
                        malicious_list = [int(n) for n in numbers]
                elif isinstance(answer, list):
                    malicious_list = answer
                else:
                    return predictions
                    
                # Mark identified packets as malicious
                for packet_num in malicious_list:
                    if 1 <= packet_num <= batch_size:
                        predictions[packet_num - 1] = True
                        
            # Handle direct list format (legacy)
            elif isinstance(data, list):
                for packet_num in data:
                    if isinstance(packet_num, int) and 1 <= packet_num <= batch_size:
                        predictions[packet_num - 1] = True
                        
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract numbers from text
            import re
            numbers = re.findall(r'\b\d+\b', response)
            for num_str in numbers:
                packet_num = int(num_str)
                if 1 <= packet_num <= batch_size:
                    predictions[packet_num - 1] = True
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response was: {response[:200]}...")
            
        return predictions
    
    def test_gpt4o(self, context: str, batch_size: int) -> Tuple[List[bool], float]:
        """Test GPT-4o on a batch of packets."""
        prompt = self.create_classification_prompt()
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": context}
        ]
        
        start_time = time.time()
        try:
            responses, usage = self.gpt4o_client.chat(messages)  # Fixed: Only 2 return values
            elapsed_time = time.time() - start_time
            
            # Track token usage
            if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                self.token_usage['gpt4o']['input_tokens'] += usage.prompt_tokens
                self.token_usage['gpt4o']['output_tokens'] += usage.completion_tokens
            
            predictions = self.parse_model_response(responses[0], batch_size)
        except Exception as e:
            print(f"Error in GPT-4o test: {e}")
            elapsed_time = time.time() - start_time
            predictions = [False] * batch_size  # Default to all benign
            
        return predictions, elapsed_time
    
    def test_minions(self, context: str, batch_size: int) -> Tuple[List[bool], float]:
        """Test Minions on a batch of packets."""
        task = self.create_classification_prompt()
        
        start_time = time.time()
        try:
            result = self.minions(
                task=task,
                doc_metadata="Network packet data",
                context=[context],
                max_rounds=5
            )
            elapsed_time = time.time() - start_time
            
            # Track token usage from Minions
            if 'remote_usage' in result:
                remote_usage = result['remote_usage']
                if 'prompt_tokens' in remote_usage:
                    self.token_usage['minions']['remote']['input_tokens'] += remote_usage['prompt_tokens']
                if 'completion_tokens' in remote_usage:
                    self.token_usage['minions']['remote']['output_tokens'] += remote_usage['completion_tokens']
            
            if 'local_usage' in result:
                local_usage = result['local_usage']
                if 'prompt_tokens' in local_usage:
                    self.token_usage['minions']['local']['input_tokens'] += local_usage['prompt_tokens']
                if 'completion_tokens' in local_usage:
                    self.token_usage['minions']['local']['output_tokens'] += local_usage['completion_tokens']
            
            # Extract predictions from final answer
            final_answer = result.get('final_answer', '')
            predictions = self.parse_model_response(final_answer, batch_size)
            
        except Exception as e:
            print(f"Error in Minions test: {e}")
            elapsed_time = time.time() - start_time
            predictions = [False] * batch_size  # Default to all benign
            
        return predictions, elapsed_time
    
    def run_comparison(self, max_batches: int = None):
        """Run the full comparison between Minions and GPT-4o."""
        print("\n" + "="*80)
        print("Starting Security Test Comparison: Minions vs GPT-4o")
        print("="*80)
        
        total_samples = len(self.df)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        # Limit batches if specified (for testing)
        if max_batches:
            num_batches = min(num_batches, max_batches)
            print(f"Limiting to {num_batches} batches for testing")
        
        all_true_labels = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            actual_batch_size = end_idx - start_idx
            
            print(f"\n--- Processing Batch {batch_idx + 1}/{num_batches} ---")
            print(f"Samples {start_idx + 1} to {end_idx}")
            
            # Prepare batch
            context, true_labels = self.prepare_packet_batch(start_idx, end_idx)
            all_true_labels.extend(true_labels)
            
            # Test GPT-4o
            print("\nTesting GPT-4o...")
            gpt4o_preds, gpt4o_time = self.test_gpt4o(context, actual_batch_size)
            self.results['gpt4o']['predictions'].extend(gpt4o_preds)
            self.results['gpt4o']['times'].append(gpt4o_time)
            
            # Print GPT-4o results for this batch
            gpt4o_malicious = sum(gpt4o_preds)
            print(f"GPT-4o identified {gpt4o_malicious}/{actual_batch_size} as malicious")
            print(f"Time taken: {gpt4o_time:.2f} seconds")
            
            # Test Minions
            print("\nTesting Minions...")
            minions_preds, minions_time = self.test_minions(context, actual_batch_size)
            self.results['minions']['predictions'].extend(minions_preds)
            self.results['minions']['times'].append(minions_time)
            
            # Print Minions results for this batch
            minions_malicious = sum(minions_preds)
            print(f"Minions identified {minions_malicious}/{actual_batch_size} as malicious")
            print(f"Time taken: {minions_time:.2f} seconds")
            
            # Batch accuracy
            true_malicious = sum(true_labels)
            print(f"\nActual malicious in batch: {true_malicious}/{actual_batch_size}")
            
            # Add a small delay between batches to avoid rate limiting
            if batch_idx < num_batches - 1:
                time.sleep(2)
        
        # Calculate final metrics
        self.calculate_metrics(all_true_labels)
    
    def calculate_metrics(self, true_labels: List[bool]):
        """Calculate and display performance metrics."""
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        metrics = {}
        
        for model_name in ['gpt4o', 'minions']:
            predictions = self.results[model_name]['predictions']
            times = self.results[model_name]['times']
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
            
            # Calculate latency metrics
            total_packets = len(predictions)
            total_time = sum(times)
            avg_time_per_batch = np.mean(times)
            avg_latency_per_packet = total_time / total_packets if total_packets > 0 else 0
            
            # Store metrics
            metrics[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'total_time': total_time,
                'avg_time_per_batch': avg_time_per_batch,
                'total_packets': total_packets,
                'avg_latency_per_packet': avg_latency_per_packet
            }
            
            # Display results
            print(f"\n{model_name.upper()} Performance:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"\n  Confusion Matrix:")
            print(f"    True Positives:  {tp}")
            print(f"    True Negatives:  {tn}")
            print(f"    False Positives: {fp}")
            print(f"    False Negatives: {fn}")
            print(f"\n  Total Time: {sum(times):.2f} seconds")
            print(f"  Avg Time per Batch: {np.mean(times):.2f} seconds")
            print(f"  Avg Latency per Packet: {metrics[model_name]['avg_latency_per_packet']:.4f} seconds")
        
        # Comparative Analysis
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS")
        print("="*80)
        
        # Accuracy comparison
        acc_diff = metrics['minions']['accuracy'] - metrics['gpt4o']['accuracy']
        print(f"\nAccuracy Difference (Minions - GPT-4o): {acc_diff:+.4f}")
        
        # Speed comparison
        if metrics['minions']['total_time'] > 0:
            speed_ratio = metrics['gpt4o']['total_time'] / metrics['minions']['total_time']
            print(f"Speed Ratio (GPT-4o time / Minions time): {speed_ratio:.2f}x")
        
        # Calculate and display costs
        costs = self.calculate_costs()
        
        print("\n" + "="*80)
        print("TOKEN USAGE & COST BREAKDOWN")
        print("="*80)
        
        print("\nGPT-4o Direct Usage:")
        print(f"  Input tokens:  {self.token_usage['gpt4o']['input_tokens']:,}")
        print(f"  Output tokens: {self.token_usage['gpt4o']['output_tokens']:,}")
        print(f"  Total cost:    ${costs['gpt4o_total']:.4f}")
        print(f"    - Input cost:  ${costs['gpt4o_input']:.4f}")
        print(f"    - Output cost: ${costs['gpt4o_output']:.4f}")
        
        print("\nMinions Token Usage:")
        print(f"  Remote (GPT-4o):")
        print(f"    Input tokens:  {self.token_usage['minions']['remote']['input_tokens']:,}")
        print(f"    Output tokens: {self.token_usage['minions']['remote']['output_tokens']:,}")
        print(f"    Cost:          ${costs['minions_remote']:.4f}")
        
        print(f"  Local (Ollama/llama3.2):")
        print(f"    Input tokens:  {self.token_usage['minions']['local']['input_tokens']:,}")
        print(f"    Output tokens: {self.token_usage['minions']['local']['output_tokens']:,}")
        if self.provider == "ollama":
            print(f"    Cost:          ${costs['minions_local']:.4f} (Free with Ollama)")
        elif self.provider == "distributed":
            print(f"    Cost:          ${costs['minions_local']:.4f} (Free with local distributed nodes)")
        else:
            print(f"    Cost:          ${costs['minions_local']:.4f}")
        
        print(f"\n  Minions Total Cost: ${costs['minions_total']:.4f}")
        
        cost_savings = costs['gpt4o_total'] - costs['minions_total']
        cost_reduction_pct = (cost_savings / costs['gpt4o_total'] * 100) if costs['gpt4o_total'] > 0 else 0
        
        print(f"\nCost Comparison:")
        print(f"  GPT-4o Direct:  ${costs['gpt4o_total']:.4f}")
        print(f"  Minions:        ${costs['minions_total']:.4f}")
        print(f"  Savings:        ${cost_savings:.4f} ({cost_reduction_pct:.1f}% reduction)")
        
        # Save results
        self.save_results(metrics, true_labels, costs)
    
    def save_results(self, metrics: Dict, true_labels: List[bool], costs: Dict):
        """Save detailed results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'total_samples': len(true_labels),
            'batch_size': self.batch_size,
            'metrics': metrics,
            'predictions': {
                'gpt4o': self.results['gpt4o']['predictions'],
                'minions': self.results['minions']['predictions']
            },
            'true_labels': true_labels,
            'token_usage': self.token_usage,
            'costs': costs
        }
        
        # Save as JSON
        output_file = f"security-test/results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {output_file}")

    def calculate_costs(self) -> Dict[str, float]:
        """Calculate costs based on token usage."""
        costs = {}
        
        # GPT-4o direct costs
        gpt4o_input_cost = self.token_usage['gpt4o']['input_tokens'] * self.pricing['gpt-4o']['input']
        gpt4o_output_cost = self.token_usage['gpt4o']['output_tokens'] * self.pricing['gpt-4o']['output']
        costs['gpt4o_total'] = gpt4o_input_cost + gpt4o_output_cost
        costs['gpt4o_input'] = gpt4o_input_cost
        costs['gpt4o_output'] = gpt4o_output_cost
        
        # Minions costs (remote is GPT-4o, local is free if using Ollama)
        minions_remote_input_cost = self.token_usage['minions']['remote']['input_tokens'] * self.pricing['gpt-4o']['input']
        minions_remote_output_cost = self.token_usage['minions']['remote']['output_tokens'] * self.pricing['gpt-4o']['output']
        
        # If using GPT-4o-mini for local, calculate its cost
        minions_local_cost = 0
        if self.provider not in ["ollama", "distributed"]:
            minions_local_input_cost = self.token_usage['minions']['local']['input_tokens'] * self.pricing['gpt-4o-mini']['input']
            minions_local_output_cost = self.token_usage['minions']['local']['output_tokens'] * self.pricing['gpt-4o-mini']['output']
            minions_local_cost = minions_local_input_cost + minions_local_output_cost
        
        costs['minions_total'] = minions_remote_input_cost + minions_remote_output_cost + minions_local_cost
        costs['minions_remote'] = minions_remote_input_cost + minions_remote_output_cost
        costs['minions_local'] = minions_local_cost
        
        return costs
    
    def print_results(self):
        """Print comprehensive results including costs."""


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Minions vs GPT-4o for security packet classification')
    parser.add_argument('--max-batches', type=int, help='Maximum number of batches to process (for testing)')
    parser.add_argument('--provider', type=str, default='ollama', choices=['ollama', 'distributed', 'openai'],
                        help='Local model provider - "ollama" (default), "distributed", or "openai" (uses gpt-4o-mini)')
    
    args = parser.parse_args()
    
    # Run the comparison
    tester = SecurityTestComparison(provider=args.provider)
    tester.run_comparison(max_batches=args.max_batches) 