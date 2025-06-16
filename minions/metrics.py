"""
Metrics collection system for Deep Research Minions
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
from contextlib import contextmanager


@dataclass
class ScrapingMetrics:
    """Detailed metrics for URL scraping operations"""
    parallel_operations: int = 0
    wall_time: float = 0.0
    cumulative_time: float = 0.0  # Sum of individual operations
    speedup_factor: float = 0.0
    per_url_times: Dict[str, float] = field(default_factory=dict)
    successful_scrapes: int = 0
    failed_scrapes: int = 0


@dataclass
class SummarizationMetrics:
    """Detailed metrics for chunk summarization"""
    batches_processed: int = 0
    average_batch_size: float = 0.0
    tokens_processed: int = 0
    time_per_batch: List[float] = field(default_factory=list)
    chunks_processed: int = 0
    relevant_chunks: int = 0


@dataclass
class RoundMetrics:
    """Metrics for a single research round"""
    round_number: int
    search_query: str
    
    # Timing
    query_generation_time: float = 0.0
    web_search_time: float = 0.0
    scraping_time: float = 0.0
    summarization_time: float = 0.0
    assessment_time: float = 0.0
    total_round_time: float = 0.0
    
    # Volume metrics
    urls_found: int = 0
    urls_scraped: int = 0
    chunks_created: int = 0
    relevant_chunks: int = 0
    
    # Detailed metrics
    scraping_details: ScrapingMetrics = field(default_factory=ScrapingMetrics)
    summarization_details: SummarizationMetrics = field(default_factory=SummarizationMetrics)
    
    # Token metrics for this round
    worker_tokens_input: int = 0
    worker_tokens_output: int = 0
    supervisor_tokens_input: int = 0
    supervisor_tokens_output: int = 0


@dataclass
class DeepResearchMetrics:
    """Overall metrics for a Deep Research run"""
    # Overall metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_time: float = 0.0
    rounds_completed: int = 0
    total_sources_processed: int = 0
    total_chunks_processed: int = 0
    
    # Per-round metrics
    rounds: List[RoundMetrics] = field(default_factory=list)
    
    # Token metrics
    total_worker_tokens_input: int = 0
    total_worker_tokens_output: int = 0
    total_supervisor_tokens_input: int = 0
    total_supervisor_tokens_output: int = 0
    worker_tokens_per_second: float = 0.0
    supervisor_tokens_per_second: float = 0.0
    
    # Efficiency metrics
    relevant_chunk_ratio: float = 0.0
    average_batch_utilization: float = 0.0
    
    # Final synthesis metrics
    synthesis_time: float = 0.0
    synthesis_tokens_input: int = 0
    synthesis_tokens_output: int = 0
    
    def finalize(self):
        """Calculate final metrics after research completes"""
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        
        # Calculate token throughput
        if self.total_time > 0:
            total_worker_tokens = self.total_worker_tokens_input + self.total_worker_tokens_output
            total_supervisor_tokens = self.total_supervisor_tokens_input + self.total_supervisor_tokens_output
            
            self.worker_tokens_per_second = total_worker_tokens / self.total_time
            self.supervisor_tokens_per_second = total_supervisor_tokens / self.total_time
        
        # Calculate efficiency metrics
        total_relevant = sum(r.relevant_chunks for r in self.rounds)
        total_chunks = sum(r.chunks_created for r in self.rounds)
        if total_chunks > 0:
            self.relevant_chunk_ratio = total_relevant / total_chunks
        
        # Calculate average batch utilization
        batch_sizes = []
        for round_metric in self.rounds:
            if round_metric.summarization_details.batches_processed > 0:
                batch_sizes.append(round_metric.summarization_details.average_batch_size)
        if batch_sizes:
            self.average_batch_utilization = sum(batch_sizes) / len(batch_sizes)


class MetricsCollector:
    """Collects and manages metrics during Deep Research execution"""
    
    def __init__(self, callback=None):
        self.metrics = DeepResearchMetrics()
        self.current_round: Optional[RoundMetrics] = None
        self.callback = callback
        self._timing_stack = []
    
    def start_round(self, round_number: int, search_query: str):
        """Start tracking a new round"""
        self.current_round = RoundMetrics(
            round_number=round_number,
            search_query=search_query
        )
        self._round_start_time = time.time()
        
        if self.callback:
            self.callback("metrics", {
                "type": "round_start",
                "round": round_number,
                "query": search_query
            })
    
    def end_round(self):
        """Finalize the current round metrics"""
        if self.current_round:
            self.current_round.total_round_time = time.time() - self._round_start_time
            self.metrics.rounds.append(self.current_round)
            self.metrics.rounds_completed += 1
            
            # Update totals
            self.metrics.total_sources_processed += self.current_round.urls_scraped
            self.metrics.total_chunks_processed += self.current_round.chunks_created
            
            # Update token counts
            self.metrics.total_worker_tokens_input += self.current_round.worker_tokens_input
            self.metrics.total_worker_tokens_output += self.current_round.worker_tokens_output
            self.metrics.total_supervisor_tokens_input += self.current_round.supervisor_tokens_input
            self.metrics.total_supervisor_tokens_output += self.current_round.supervisor_tokens_output
            
            if self.callback:
                self.callback("metrics", {
                    "type": "round_complete",
                    "round": self.current_round.round_number,
                    "time": self.current_round.total_round_time,
                    "sources": self.current_round.urls_scraped,
                    "chunks": self.current_round.chunks_created,
                    "relevant": self.current_round.relevant_chunks
                })
            
            self.current_round = None
    
    @contextmanager
    def measure_phase(self, phase_name: str):
        """Context manager to measure time for a specific phase"""
        start_time = time.time()
        self._timing_stack.append((phase_name, start_time))
        
        try:
            yield
        finally:
            phase_name, start_time = self._timing_stack.pop()
            elapsed = time.time() - start_time
            
            if self.current_round:
                if phase_name == "query_generation":
                    self.current_round.query_generation_time = elapsed
                elif phase_name == "web_search":
                    self.current_round.web_search_time = elapsed
                elif phase_name == "scraping":
                    self.current_round.scraping_time = elapsed
                elif phase_name == "summarization":
                    self.current_round.summarization_time = elapsed
                elif phase_name == "assessment":
                    self.current_round.assessment_time = elapsed
            elif phase_name == "synthesis":
                self.metrics.synthesis_time = elapsed
            
            if self.callback:
                self.callback("metrics", {
                    "type": "phase_complete",
                    "phase": phase_name,
                    "time": elapsed
                })
    
    def update_scraping_metrics(self, 
                              urls_found: int,
                              per_url_times: Dict[str, float],
                              successful: int,
                              failed: int):
        """Update scraping metrics for the current round"""
        if self.current_round:
            self.current_round.urls_found = urls_found
            self.current_round.urls_scraped = successful
            
            details = self.current_round.scraping_details
            details.parallel_operations = len(per_url_times)
            details.per_url_times = per_url_times
            details.successful_scrapes = successful
            details.failed_scrapes = failed
            
            # Calculate timing metrics
            if per_url_times:
                details.cumulative_time = sum(per_url_times.values())
                details.wall_time = self.current_round.scraping_time
                if details.wall_time > 0:
                    details.speedup_factor = details.cumulative_time / details.wall_time
    
    def update_summarization_batch(self, 
                                 batch_size: int,
                                 batch_time: float,
                                 chunks_in_batch: int,
                                 relevant_in_batch: int):
        """Update metrics for a summarization batch"""
        if self.current_round:
            details = self.current_round.summarization_details
            details.batches_processed += 1
            details.time_per_batch.append(batch_time)
            details.chunks_processed += chunks_in_batch
            details.relevant_chunks += relevant_in_batch
            
            # Update round totals
            self.current_round.chunks_created += chunks_in_batch
            self.current_round.relevant_chunks += relevant_in_batch
            
            # Recalculate average batch size
            total_chunks = sum(self.current_round.summarization_details.time_per_batch)
            if details.batches_processed > 0:
                details.average_batch_size = details.chunks_processed / details.batches_processed
            
            if self.callback:
                self.callback("metrics", {
                    "type": "batch_complete",
                    "batch_num": details.batches_processed,
                    "batch_size": batch_size,
                    "time": batch_time,
                    "relevant": relevant_in_batch
                })
    
    def update_token_usage(self, 
                         model_type: str,
                         input_tokens: int,
                         output_tokens: int):
        """Update token usage metrics"""
        if model_type == "worker":
            if self.current_round:
                self.current_round.worker_tokens_input += input_tokens
                self.current_round.worker_tokens_output += output_tokens
            else:
                # Must be synthesis phase
                self.metrics.synthesis_tokens_input += input_tokens
                self.metrics.synthesis_tokens_output += output_tokens
        elif model_type == "supervisor":
            if self.current_round:
                self.current_round.supervisor_tokens_input += input_tokens
                self.current_round.supervisor_tokens_output += output_tokens
        
        # Calculate current throughput
        elapsed = time.time() - self.metrics.start_time
        if elapsed > 0:
            total_worker = (self.metrics.total_worker_tokens_input + 
                          self.metrics.total_worker_tokens_output)
            total_supervisor = (self.metrics.total_supervisor_tokens_input + 
                              self.metrics.total_supervisor_tokens_output)
            
            if self.callback:
                self.callback("metrics", {
                    "type": "throughput_update",
                    "worker_tps": total_worker / elapsed,
                    "supervisor_tps": total_supervisor / elapsed,
                    "elapsed_time": elapsed
                })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        self.metrics.finalize()
        
        return {
            "total_time": self.metrics.total_time,
            "rounds_completed": self.metrics.rounds_completed,
            "total_sources": self.metrics.total_sources_processed,
            "total_chunks": self.metrics.total_chunks_processed,
            "relevant_ratio": f"{self.metrics.relevant_chunk_ratio:.1%}",
            "worker_throughput": f"{self.metrics.worker_tokens_per_second:.0f} tokens/sec",
            "supervisor_throughput": f"{self.metrics.supervisor_tokens_per_second:.0f} tokens/sec",
            "rounds": [
                {
                    "number": r.round_number,
                    "query": r.search_query,
                    "time": r.total_round_time,
                    "sources": r.urls_scraped,
                    "chunks": r.chunks_created,
                    "relevant": r.relevant_chunks,
                    "phases": {
                        "query_gen": r.query_generation_time,
                        "web_search": r.web_search_time,
                        "scraping": r.scraping_time,
                        "summarization": r.summarization_time,
                        "assessment": r.assessment_time
                    }
                }
                for r in self.metrics.rounds
            ],
            "synthesis_time": self.metrics.synthesis_time
        } 