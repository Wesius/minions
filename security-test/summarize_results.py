import json
import os
import glob
from datetime import datetime
from typing import Dict, List
import pandas as pd

def load_results_files(directory: str = "security-test") -> List[Dict]:
    """Load all results JSON files from the directory."""
    pattern = os.path.join(directory, "results_*.json")
    files = glob.glob(pattern)
    
    results = []
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                data['filename'] = os.path.basename(file)
                results.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return results

def summarize_results(results: List[Dict]) -> None:
    """Create a comprehensive summary of all test results."""
    
    if not results:
        print("No results files found!")
        return
    
    print("="*80)
    print("SECURITY TEST RESULTS SUMMARY")
    print("="*80)
    print(f"\nTotal test runs analyzed: {len(results)}")
    
    # Collect metrics for each model
    gpt4o_metrics = []
    minions_metrics = []
    
    for result in results:
        timestamp = result.get('timestamp', 'Unknown')
        filename = result.get('filename', 'Unknown')
        
        print(f"\n--- Test Run: {filename} ---")
        print(f"Timestamp: {timestamp}")
        print(f"Total samples: {result.get('total_samples', 'N/A')}")
        print(f"Batch size: {result.get('batch_size', 'N/A')}")
        
        # Extract metrics
        if 'metrics' in result:
            metrics = result['metrics']
            
            # GPT-4o metrics
            if 'gpt4o' in metrics:
                gpt4o = metrics['gpt4o']
                gpt4o['run'] = filename
                gpt4o_metrics.append(gpt4o)
                
                print(f"\nGPT-4o Performance:")
                print(f"  Accuracy: {gpt4o.get('accuracy', 0):.4f}")
                print(f"  Time: {gpt4o.get('total_time', 0):.2f}s")
                if 'avg_latency_per_packet' in gpt4o:
                    print(f"  Latency per packet: {gpt4o.get('avg_latency_per_packet', 0):.4f}s")
                elif 'total_packets' in gpt4o and gpt4o.get('total_packets', 0) > 0:
                    # Calculate if not present
                    latency = gpt4o.get('total_time', 0) / gpt4o.get('total_packets', 1)
                    print(f"  Latency per packet: {latency:.4f}s")
            
            # Minions metrics
            if 'minions' in metrics:
                minions = metrics['minions']
                minions['run'] = filename
                minions_metrics.append(minions)
                
                print(f"\nMinions Performance:")
                print(f"  Accuracy: {minions.get('accuracy', 0):.4f}")
                print(f"  Time: {minions.get('total_time', 0):.2f}s")
                if 'avg_latency_per_packet' in minions:
                    print(f"  Latency per packet: {minions.get('avg_latency_per_packet', 0):.4f}s")
                elif 'total_packets' in minions and minions.get('total_packets', 0) > 0:
                    # Calculate if not present
                    latency = minions.get('total_time', 0) / minions.get('total_packets', 1)
                    print(f"  Latency per packet: {latency:.4f}s")
        
        # Cost information
        if 'costs' in result:
            costs = result['costs']
            print(f"\nCost Analysis:")
            print(f"  GPT-4o: ${costs.get('gpt4o_total', 0):.4f}")
            print(f"  Minions: ${costs.get('minions_total', 0):.4f}")
            savings = costs.get('gpt4o_total', 0) - costs.get('minions_total', 0)
            if costs.get('gpt4o_total', 0) > 0:
                savings_pct = (savings / costs['gpt4o_total']) * 100
                print(f"  Savings: ${savings:.4f} ({savings_pct:.1f}%)")
        
        # Token usage
        if 'token_usage' in result:
            tokens = result['token_usage']
            print(f"\nToken Usage:")
            
            if 'gpt4o' in tokens:
                gpt4o_tokens = tokens['gpt4o']
                print(f"  GPT-4o:")
                print(f"    Input: {gpt4o_tokens.get('input_tokens', 0):,}")
                print(f"    Output: {gpt4o_tokens.get('output_tokens', 0):,}")
            
            if 'minions' in tokens:
                minions_tokens = tokens['minions']
                print(f"  Minions:")
                print(f"    Remote (GPT-4o):")
                print(f"      Input: {minions_tokens.get('remote', {}).get('input_tokens', 0):,}")
                print(f"      Output: {minions_tokens.get('remote', {}).get('output_tokens', 0):,}")
                print(f"    Local (Ollama):")
                print(f"      Input: {minions_tokens.get('local', {}).get('input_tokens', 0):,}")
                print(f"      Output: {minions_tokens.get('local', {}).get('output_tokens', 0):,}")
    
    # Calculate aggregate statistics
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)
    
    if gpt4o_metrics:
        print("\nGPT-4o Average Performance:")
        avg_accuracy = sum(m.get('accuracy', 0) for m in gpt4o_metrics) / len(gpt4o_metrics)
        avg_time = sum(m.get('total_time', 0) for m in gpt4o_metrics) / len(gpt4o_metrics)
        
        # Calculate average latency
        latencies = []
        for m in gpt4o_metrics:
            if 'avg_latency_per_packet' in m:
                latencies.append(m['avg_latency_per_packet'])
            elif 'total_packets' in m and m.get('total_packets', 0) > 0:
                latencies.append(m.get('total_time', 0) / m.get('total_packets', 1))
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        print(f"  Accuracy: {avg_accuracy:.4f}")
        print(f"  Avg Time: {avg_time:.2f}s")
        print(f"  Avg Latency per Packet: {avg_latency:.4f}s")
    
    if minions_metrics:
        print("\nMinions Average Performance:")
        avg_accuracy = sum(m.get('accuracy', 0) for m in minions_metrics) / len(minions_metrics)
        avg_time = sum(m.get('total_time', 0) for m in minions_metrics) / len(minions_metrics)
        
        # Calculate average latency
        latencies = []
        for m in minions_metrics:
            if 'avg_latency_per_packet' in m:
                latencies.append(m['avg_latency_per_packet'])
            elif 'total_packets' in m and m.get('total_packets', 0) > 0:
                latencies.append(m.get('total_time', 0) / m.get('total_packets', 1))
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        print(f"  Accuracy: {avg_accuracy:.4f}")
        print(f"  Avg Time: {avg_time:.2f}s")
        print(f"  Avg Latency per Packet: {avg_latency:.4f}s")
    
    # Create comparison table
    if gpt4o_metrics and minions_metrics:
        print("\n" + "="*80)
        print("DETAILED COMPARISON TABLE")
        print("="*80)
        
        # Create DataFrame for better visualization
        comparison_data = []
        for i, (g, m) in enumerate(zip(gpt4o_metrics, minions_metrics)):
            # Calculate latencies if not present
            g_latency = g.get('avg_latency_per_packet', 0)
            if g_latency == 0 and 'total_packets' in g and g.get('total_packets', 0) > 0:
                g_latency = g.get('total_time', 0) / g.get('total_packets', 1)
            
            m_latency = m.get('avg_latency_per_packet', 0)
            if m_latency == 0 and 'total_packets' in m and m.get('total_packets', 0) > 0:
                m_latency = m.get('total_time', 0) / m.get('total_packets', 1)
            
            comparison_data.append({
                'Run': i + 1,
                'GPT4o_Acc': f"{g.get('accuracy', 0):.3f}",
                'Minions_Acc': f"{m.get('accuracy', 0):.3f}",
                'GPT4o_Time': f"{g.get('total_time', 0):.1f}s",
                'Minions_Time': f"{m.get('total_time', 0):.1f}s",
                'GPT4o_Latency': f"{g_latency:.3f}s",
                'Minions_Latency': f"{m_latency:.3f}s",
                'Acc_Diff': f"{(m.get('accuracy', 0) - g.get('accuracy', 0)):.3f}",
            })
        
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
    
    # Cost summary
    print("\n" + "="*80)
    print("COST EFFICIENCY SUMMARY")
    print("="*80)
    
    total_gpt4o_cost = 0
    total_minions_cost = 0
    
    for result in results:
        if 'costs' in result:
            costs = result['costs']
            total_gpt4o_cost += costs.get('gpt4o_total', 0)
            total_minions_cost += costs.get('minions_total', 0)
    
    if total_gpt4o_cost > 0:
        total_savings = total_gpt4o_cost - total_minions_cost
        savings_pct = (total_savings / total_gpt4o_cost) * 100
        
        print(f"\nTotal costs across all runs:")
        print(f"  GPT-4o Direct: ${total_gpt4o_cost:.4f}")
        print(f"  Minions: ${total_minions_cost:.4f}")
        print(f"  Total Savings: ${total_savings:.4f} ({savings_pct:.1f}%)")
    


def save_summary(results: List[Dict]) -> None:
    """Save summary to a text file."""
    summary_lines = []
    
    if not results:
        return
    
    summary_lines.append("="*80)
    summary_lines.append("SECURITY TEST RESULTS SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append(f"\nTotal test runs analyzed: {len(results)}")
    
    # Collect metrics
    gpt4o_metrics = []
    minions_metrics = []
    
    for result in results:
        if 'metrics' in result:
            metrics = result['metrics']
            if 'gpt4o' in metrics:
                gpt4o_metrics.append(metrics['gpt4o'])
            if 'minions' in metrics:
                minions_metrics.append(metrics['minions'])
    
    # Aggregate statistics
    if gpt4o_metrics:
        summary_lines.append("\nGPT-4o Average Performance:")
        avg_accuracy = sum(m.get('accuracy', 0) for m in gpt4o_metrics) / len(gpt4o_metrics)
        summary_lines.append(f"  Accuracy: {avg_accuracy:.4f}")
    
    if minions_metrics:
        summary_lines.append("\nMinions Average Performance:")
        avg_accuracy = sum(m.get('accuracy', 0) for m in minions_metrics) / len(minions_metrics)
        summary_lines.append(f"  Accuracy: {avg_accuracy:.4f}")
    
    # Cost summary
    total_gpt4o_cost = 0
    total_minions_cost = 0
    
    for result in results:
        if 'costs' in result:
            costs = result['costs']
            total_gpt4o_cost += costs.get('gpt4o_total', 0)
            total_minions_cost += costs.get('minions_total', 0)
    
    if total_gpt4o_cost > 0:
        total_savings = total_gpt4o_cost - total_minions_cost
        savings_pct = (total_savings / total_gpt4o_cost) * 100
        
        summary_lines.append(f"\nTotal Cost Summary:")
        summary_lines.append(f"  GPT-4o Direct: ${total_gpt4o_cost:.4f}")
        summary_lines.append(f"  Minions: ${total_minions_cost:.4f}")
        summary_lines.append(f"  Total Savings: ${total_savings:.4f} ({savings_pct:.1f}%)")
    
    # Save to file
    summary_file = os.path.join("security-test", f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    results = load_results_files()
    summarize_results(results)
    save_summary(results) 