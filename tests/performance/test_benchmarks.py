"""Performance benchmark tests."""

import pytest
import time
import statistics
from typing import List
import numpy as np
import pandas as pd

# Import application modules for benchmarking
try:
    from src.association_engine import AssociationEngine
    from src.layout_optimizer import LayoutOptimizer
    from src.qloo_client import QlooClient
except ImportError:
    pytest.skip("Application modules not available", allow_module_level=True)


class TestAssociationPerformance:
    """Performance tests for association mining."""
    
    @pytest.fixture
    def sample_transactions(self):
        """Generate sample transaction data for testing."""
        np.random.seed(42)
        products = [f"product_{i}" for i in range(100)]
        transactions = []
        
        for _ in range(1000):
            transaction_size = np.random.randint(2, 10)
            transaction = np.random.choice(products, transaction_size, replace=False).tolist()
            transactions.append(transaction)
        
        return transactions
    
    @pytest.fixture
    def large_transactions(self):
        """Generate large transaction dataset for stress testing."""
        np.random.seed(42)
        products = [f"product_{i}" for i in range(500)]
        transactions = []
        
        for _ in range(10000):
            transaction_size = np.random.randint(2, 15)
            transaction = np.random.choice(products, transaction_size, replace=False).tolist()
            transactions.append(transaction)
        
        return transactions
    
    def test_association_mining_performance(self, sample_transactions, benchmark):
        """Benchmark association mining performance."""
        engine = AssociationEngine(min_support=0.02, min_confidence=0.5)
        engine.load_transactions(sample_transactions)
        
        def mine_associations():
            return engine.mine_associations()
        
        result = benchmark(mine_associations)
        
        # Verify results are meaningful
        assert len(result) > 0
        assert all(hasattr(rule, 'confidence') for rule in result)
    
    def test_frequent_itemsets_performance(self, sample_transactions, benchmark):
        """Benchmark frequent itemset mining performance."""
        engine = AssociationEngine(min_support=0.02)
        engine.load_transactions(sample_transactions)
        
        def mine_itemsets():
            return engine.mine_frequent_itemsets()
        
        result = benchmark(mine_itemsets)
        assert len(result) > 0
    
    @pytest.mark.slow
    def test_large_dataset_performance(self, large_transactions):
        """Test performance with large datasets."""
        engine = AssociationEngine(min_support=0.001, min_confidence=0.3)
        
        # Time the loading
        start_time = time.time()
        engine.load_transactions(large_transactions)
        load_time = time.time() - start_time
        
        # Time the mining
        start_time = time.time()
        rules = engine.mine_associations()
        mining_time = time.time() - start_time
        
        # Performance assertions
        assert load_time < 5.0  # Should load within 5 seconds
        assert mining_time < 30.0  # Should mine within 30 seconds
        assert len(rules) > 0
        
        print(f"Large dataset performance:")
        print(f"  Load time: {load_time:.2f}s")
        print(f"  Mining time: {mining_time:.2f}s")
        print(f"  Rules found: {len(rules)}")
    
    def test_memory_usage_during_mining(self, sample_transactions):
        """Test memory usage during association mining."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = AssociationEngine(min_support=0.02, min_confidence=0.5)
        engine.load_transactions(sample_transactions)
        
        load_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        rules = engine.mine_associations()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 100  # Should not use more than 100MB for sample data
        
        print(f"Memory usage:")
        print(f"  Initial: {initial_memory:.1f}MB")
        print(f"  After load: {load_memory:.1f}MB") 
        print(f"  After mining: {final_memory:.1f}MB")
        print(f"  Total increase: {memory_increase:.1f}MB")


class TestLayoutOptimizerPerformance:
    """Performance tests for layout optimization."""
    
    @pytest.fixture
    def sample_layout_data(self):
        """Generate sample layout optimization data."""
        products = [f"product_{i}" for i in range(50)]
        sections = [f"section_{i}" for i in range(10)]
        
        # Generate random association strengths
        associations = {}
        np.random.seed(42)
        
        for i, product1 in enumerate(products):
            associations[product1] = {}
            for product2 in products[i+1:i+6]:  # Limit associations for performance
                associations[product1][product2] = np.random.random()
        
        return {
            'products': products,
            'sections': sections,
            'associations': associations
        }
    
    def test_optimization_algorithm_performance(self, sample_layout_data, benchmark):
        """Benchmark layout optimization algorithm."""
        optimizer = LayoutOptimizer()
        
        def run_optimization():
            return optimizer.optimize_layout(
                products=sample_layout_data['products'][:20],  # Limit for benchmark
                sections=sample_layout_data['sections'][:5],
                associations=sample_layout_data['associations']
            )
        
        result = benchmark(run_optimization)
        
        # Verify result is meaningful
        assert result is not None
        assert hasattr(result, 'score') or isinstance(result, dict)
    
    @pytest.mark.slow
    def test_large_optimization_performance(self, sample_layout_data):
        """Test optimization performance with larger datasets."""
        optimizer = LayoutOptimizer()
        
        start_time = time.time()
        result = optimizer.optimize_layout(
            products=sample_layout_data['products'],
            sections=sample_layout_data['sections'],
            associations=sample_layout_data['associations']
        )
        optimization_time = time.time() - start_time
        
        # Performance assertions
        assert optimization_time < 60.0  # Should complete within 1 minute
        assert result is not None
        
        print(f"Large optimization performance:")
        print(f"  Optimization time: {optimization_time:.2f}s")
        print(f"  Products optimized: {len(sample_layout_data['products'])}")
        print(f"  Sections used: {len(sample_layout_data['sections'])}")
    
    def test_optimization_convergence_speed(self, sample_layout_data):
        """Test how quickly optimization converges."""
        optimizer = LayoutOptimizer()
        
        # Run optimization with different iteration limits
        iteration_limits = [10, 50, 100, 500]
        times = []
        scores = []
        
        for limit in iteration_limits:
            start_time = time.time()
            result = optimizer.optimize_layout(
                products=sample_layout_data['products'][:15],
                sections=sample_layout_data['sections'][:3],
                associations=sample_layout_data['associations'],
                max_iterations=limit
            )
            duration = time.time() - start_time
            
            times.append(duration)
            scores.append(getattr(result, 'score', 0))
        
        # Verify convergence behavior
        assert all(t > 0 for t in times)
        assert times[-1] > times[0]  # More iterations should take longer
        
        print("Convergence analysis:")
        for i, (limit, time_taken, score) in enumerate(zip(iteration_limits, times, scores)):
            print(f"  {limit:3d} iterations: {time_taken:.3f}s, score: {score:.3f}")


class TestQlooClientPerformance:
    """Performance tests for Qloo API client."""
    
    @pytest.fixture
    def client(self):
        """Create QlooClient instance."""
        return QlooClient()
    
    def test_api_request_performance(self, client, benchmark):
        """Benchmark API request performance."""
        def make_request():
            return client.get_product_recommendations("apple", limit=5)
        
        result = benchmark(make_request)
        
        # Verify result structure
        assert isinstance(result, list) or result is None
    
    def test_concurrent_api_requests(self, client):
        """Test performance of concurrent API requests."""
        import concurrent.futures
        import threading
        
        def make_request(query):
            start_time = time.time()
            result = client.get_product_recommendations(query, limit=3)
            return time.time() - start_time, result
        
        queries = ["apple", "milk", "bread", "eggs", "cheese"]
        
        # Sequential requests
        start_time = time.time()
        sequential_results = []
        for query in queries:
            duration, result = make_request(query)
            sequential_results.append(duration)
        sequential_total = time.time() - start_time
        
        # Concurrent requests
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, query) for query in queries]
            concurrent_results = [future.result()[0] for future in concurrent.futures.as_completed(futures)]
        concurrent_total = time.time() - start_time
        
        # Performance analysis
        avg_sequential = statistics.mean(sequential_results)
        avg_concurrent = statistics.mean(concurrent_results)
        
        print(f"API request performance:")
        print(f"  Sequential total: {sequential_total:.2f}s")
        print(f"  Concurrent total: {concurrent_total:.2f}s")
        print(f"  Average sequential: {avg_sequential:.2f}s")
        print(f"  Average concurrent: {avg_concurrent:.2f}s")
        
        # Concurrent should be faster overall (despite potential rate limiting)
        # But individual requests might be slower due to contention
        assert concurrent_total < sequential_total * 1.5  # Allow some overhead
    
    def test_api_response_time_consistency(self, client):
        """Test consistency of API response times."""
        query = "test"
        response_times = []
        
        # Make multiple requests to measure consistency
        for _ in range(10):
            start_time = time.time()
            client.get_product_recommendations(query, limit=3)
            response_time = time.time() - start_time
            response_times.append(response_time)
            time.sleep(0.5)  # Small delay between requests
        
        # Calculate statistics
        mean_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        
        print(f"API response time consistency:")
        print(f"  Mean: {mean_time:.3f}s")
        print(f"  Median: {median_time:.3f}s")
        print(f"  Std Dev: {std_dev:.3f}s")
        print(f"  Min: {min(response_times):.3f}s")
        print(f"  Max: {max(response_times):.3f}s")
        
        # Response times should be reasonably consistent
        assert std_dev < mean_time * 0.5  # Standard deviation should be less than 50% of mean
        assert max(response_times) < mean_time * 3  # No outliers beyond 3x mean


class TestDataProcessingPerformance:
    """Performance tests for data processing operations."""
    
    def test_dataframe_operations_performance(self, benchmark):
        """Benchmark pandas DataFrame operations."""
        # Generate large dataset
        np.random.seed(42)
        data = {
            'product_id': [f"prod_{i}" for i in range(10000)],
            'category': np.random.choice(['A', 'B', 'C', 'D'], 10000),
            'price': np.random.uniform(1, 100, 10000),
            'sales': np.random.randint(1, 1000, 10000)
        }
        
        df = pd.DataFrame(data)
        
        def process_data():
            # Typical data processing operations
            result = df.groupby('category').agg({
                'price': ['mean', 'std'],
                'sales': ['sum', 'count']
            }).round(2)
            
            # Add some filtering and sorting
            filtered = df[df['price'] > 50]
            sorted_df = filtered.sort_values('sales', ascending=False)
            
            return result, sorted_df.head(100)
        
        result = benchmark(process_data)
        
        # Verify results
        assert len(result) == 2
        assert not result[0].empty
        assert not result[1].empty
    
    def test_numpy_computations_performance(self, benchmark):
        """Benchmark NumPy computation performance."""
        np.random.seed(42)
        
        # Generate large arrays
        array1 = np.random.random((1000, 1000))
        array2 = np.random.random((1000, 1000))
        
        def compute_operations():
            # Matrix operations
            dot_product = np.dot(array1, array2)
            
            # Statistical operations
            mean_vals = np.mean(array1, axis=0)
            std_vals = np.std(array1, axis=0)
            
            # Element-wise operations
            result = array1 * array2 + np.sqrt(array1)
            
            return dot_product, mean_vals, std_vals, result
        
        result = benchmark(compute_operations)
        
        # Verify results
        assert len(result) == 4
        assert all(isinstance(r, np.ndarray) for r in result)


# Configuration for pytest-benchmark
def pytest_configure(config):
    """Configure pytest-benchmark."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (may take several seconds to minutes)"
    )