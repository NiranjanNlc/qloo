"""
Optimizer Performance Benchmark Suite

This module provides comprehensive benchmarking capabilities for the
supermarket layout optimizer, testing runtime performance with large
datasets and profiling performance hotspots.
"""

import time
import cProfile
import pstats
import io
import memory_profiler
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from layout_optimizer import SupermarketLayoutOptimizer, SectionLevelOptimizer
    from optimization_heuristics import HeuristicOptimizer
    from association_engine import AprioriAssociationEngine
    from algorithms.scoring import ScoringsEngine, ScoreType
    from models import Product, Combo
    from qloo_client import create_qloo_client
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizerBenchmark:
    """Main benchmark suite for optimizer performance testing."""

    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark suite.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results = {}
        self.profiling_data = {}

        # Performance targets (in seconds)
        self.targets = {
            "1k_skus": 10.0,
            "5k_skus": 60.0,
            "10k_skus": 180.0,  # Target: under 180 seconds for 10k SKUs
            "15k_skus": 300.0,
        }

    def generate_mock_data(
        self, num_skus: int, num_transactions: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate mock product catalog and transaction data for benchmarking.

        Args:
            num_skus: Number of SKUs to generate
            num_transactions: Number of transactions (default: num_skus * 10)

        Returns:
            Tuple of (catalog_df, transactions_df)
        """
        if num_transactions is None:
            num_transactions = num_skus * 10

        logger.info(
            f"Generating mock data: {num_skus} SKUs, {num_transactions} transactions"
        )

        # Generate product catalog
        np.random.seed(42)  # For reproducible results

        categories = [
            "Dairy",
            "Produce",
            "Meat",
            "Beverages",
            "Snacks",
            "Bakery",
            "Frozen",
            "Canned",
            "Personal Care",
            "Household",
        ]

        catalog_data = []
        for i in range(1, num_skus + 1):
            catalog_data.append(
                {
                    "product_id": i,
                    "product_name": f"Product {i}",
                    "category": np.random.choice(categories),
                    "price": round(np.random.uniform(0.99, 19.99), 2),
                    "weight": round(np.random.uniform(0.1, 5.0), 2),
                    "brand": f"Brand {i % 50}",  # 50 different brands
                    "seasonal": np.random.choice([True, False], p=[0.2, 0.8]),
                }
            )

        catalog_df = pd.DataFrame(catalog_data)

        # Generate transaction data with realistic patterns
        transaction_data = []

        # Create product popularity distribution (80/20 rule)
        popular_products = list(range(1, int(num_skus * 0.2) + 1))
        regular_products = list(range(int(num_skus * 0.2) + 1, num_skus + 1))

        for transaction_id in range(1, num_transactions + 1):
            # Random basket size (realistic distribution)
            basket_size = max(1, int(np.random.gamma(2, 2)))  # Average ~4 items
            basket_size = min(basket_size, 20)  # Cap at 20 items

            # Select products with popularity bias
            products_in_basket = []

            for _ in range(basket_size):
                if np.random.random() < 0.7:  # 70% chance of popular product
                    product = np.random.choice(popular_products)
                else:
                    product = np.random.choice(regular_products)

                if product not in products_in_basket:
                    products_in_basket.append(product)

            # Add transaction records
            for product_id in products_in_basket:
                transaction_data.append(
                    {
                        "transaction_id": transaction_id,
                        "product_id": product_id,
                        "quantity": np.random.randint(1, 4),
                        "timestamp": datetime.now()
                        - timedelta(days=np.random.randint(0, 90)),
                    }
                )

        transactions_df = pd.DataFrame(transaction_data)

        logger.info(
            f"Generated {len(catalog_df)} products and {len(transactions_df)} transaction records"
        )
        return catalog_df, transactions_df

    @memory_profiler.profile
    def benchmark_optimizer_initialization(
        self, catalog_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Benchmark optimizer initialization performance."""
        logger.info("Benchmarking optimizer initialization...")

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Initialize optimizer
        optimizer = SupermarketLayoutOptimizer()
        optimizer.load_data(catalog_df, transactions_df)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        result = {
            "initialization_time": end_time - start_time,
            "memory_usage_mb": end_memory - start_memory,
            "num_products": len(catalog_df),
            "num_transactions": len(transactions_df),
        }

        logger.info(
            f"Initialization completed in {result['initialization_time']:.2f}s, memory: {result['memory_usage_mb']:.1f}MB"
        )
        return result

    def benchmark_association_engine(
        self, transactions_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Benchmark association rule mining performance."""
        logger.info("Benchmarking association engine...")

        # Convert transactions to required format
        transaction_matrix = self._convert_to_transaction_matrix(transactions_df)

        start_time = time.time()

        # Initialize and train association engine
        engine = AprioriAssociationEngine()
        engine.train(transaction_matrix)

        training_time = time.time() - start_time

        # Test rule generation
        start_time = time.time()
        rules = engine.association_rules
        rule_generation_time = time.time() - start_time

        # Test association lookups
        start_time = time.time()
        sample_products = list(range(1, min(100, len(transaction_matrix.columns))))
        for product_id in sample_products:
            associations = engine.get_associations(product_id, top_k=10)
        lookup_time = time.time() - start_time

        result = {
            "training_time": training_time,
            "rule_generation_time": rule_generation_time,
            "lookup_time": lookup_time,
            "total_rules": len(rules),
            "num_products": len(transaction_matrix.columns),
            "matrix_size": transaction_matrix.shape,
        }

        logger.info(
            f"Association engine: training {training_time:.2f}s, {len(rules)} rules generated"
        )
        return result

    def benchmark_layout_optimization(
        self, optimizer: SupermarketLayoutOptimizer
    ) -> Dict[str, Any]:
        """Benchmark layout optimization performance."""
        logger.info("Benchmarking layout optimization...")

        start_time = time.time()

        # Run layout optimization with different goals
        optimization_goals = [
            "maximize_associations",
            "improve_flow",
            "category_adjacency",
        ]
        recommendations = optimizer.optimize_layout(optimization_goals)

        optimization_time = time.time() - start_time

        # Test section-level optimization
        start_time = time.time()
        section_optimizer = SectionLevelOptimizer(optimizer)
        section_results = section_optimizer.optimize_store_sections()
        section_optimization_time = time.time() - start_time

        result = {
            "layout_optimization_time": optimization_time,
            "section_optimization_time": section_optimization_time,
            "total_recommendations": len(recommendations),
            "section_results": len(section_results),
            "optimization_goals": len(optimization_goals),
        }

        logger.info(
            f"Layout optimization: {optimization_time:.2f}s, {len(recommendations)} recommendations"
        )
        return result

    def benchmark_heuristic_scoring(
        self, optimizer: SupermarketLayoutOptimizer
    ) -> Dict[str, Any]:
        """Benchmark heuristic scoring performance."""
        logger.info("Benchmarking heuristic scoring...")

        # Initialize heuristic optimizer
        heuristic_optimizer = HeuristicOptimizer(optimizer)

        start_time = time.time()

        # Calculate heuristic scores for all products
        scores = heuristic_optimizer.calculate_heuristic_scores()

        scoring_time = time.time() - start_time

        # Test placement suggestions
        start_time = time.time()
        suggestions = heuristic_optimizer.generate_placement_suggestions()
        suggestions_time = time.time() - start_time

        result = {
            "heuristic_scoring_time": scoring_time,
            "suggestions_generation_time": suggestions_time,
            "total_scores": len(scores),
            "total_suggestions": len(suggestions),
            "avg_score": np.mean([s.final_score for s in scores]) if scores else 0,
        }

        logger.info(
            f"Heuristic scoring: {scoring_time:.2f}s, {len(scores)} scores calculated"
        )
        return result

    def profile_optimization_hotspots(self, num_skus: int) -> Dict[str, Any]:
        """Profile optimization performance to identify hotspots."""
        logger.info(f"Profiling optimization hotspots for {num_skus} SKUs...")

        # Generate data
        catalog_df, transactions_df = self.generate_mock_data(num_skus)

        # Create profiler
        profiler = cProfile.Profile()

        # Profile the full optimization process
        profiler.enable()

        try:
            # Initialize optimizer
            optimizer = SupermarketLayoutOptimizer()
            optimizer.load_data(catalog_df, transactions_df)

            # Run optimization
            recommendations = optimizer.optimize_layout(["maximize_associations"])

            # Run heuristic scoring
            heuristic_optimizer = HeuristicOptimizer(optimizer)
            scores = heuristic_optimizer.calculate_heuristic_scores()

        finally:
            profiler.disable()

        # Collect profiling statistics
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions

        # Parse profiling data
        profiling_output = stats_buffer.getvalue()

        # Save detailed profiling data
        profile_file = self.output_dir / f"profile_{num_skus}_skus.prof"
        profiler.dump_stats(str(profile_file))

        # Extract hotspots
        hotspots = self._extract_hotspots(stats)

        result = {
            "num_skus": num_skus,
            "profile_file": str(profile_file),
            "hotspots": hotspots,
            "profiling_output": profiling_output[:2000],  # First 2000 chars
        }

        return result

    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite with different SKU counts."""
        logger.info("Starting full benchmark suite...")

        sku_counts = [1000, 5000, 10000, 15000]
        benchmark_results = {}

        for num_skus in sku_counts:
            logger.info(f"\n{'='*50}")
            logger.info(f"BENCHMARKING {num_skus} SKUs")
            logger.info(f"{'='*50}")

            try:
                # Generate test data
                catalog_df, transactions_df = self.generate_mock_data(num_skus)

                # Track overall timing
                overall_start = time.time()

                # Individual benchmarks
                init_results = self.benchmark_optimizer_initialization(
                    catalog_df, transactions_df
                )

                # Re-initialize for subsequent tests
                optimizer = SupermarketLayoutOptimizer()
                optimizer.load_data(catalog_df, transactions_df)

                assoc_results = self.benchmark_association_engine(transactions_df)
                layout_results = self.benchmark_layout_optimization(optimizer)
                heuristic_results = self.benchmark_heuristic_scoring(optimizer)

                overall_time = time.time() - overall_start

                # Profile hotspots for this SKU count
                profiling_results = self.profile_optimization_hotspots(num_skus)

                # Combine results
                sku_key = f"{num_skus//1000}k_skus"
                benchmark_results[sku_key] = {
                    "num_skus": num_skus,
                    "overall_time": overall_time,
                    "target_time": self.targets.get(sku_key, float("inf")),
                    "meets_target": overall_time
                    <= self.targets.get(sku_key, float("inf")),
                    "initialization": init_results,
                    "association_engine": assoc_results,
                    "layout_optimization": layout_results,
                    "heuristic_scoring": heuristic_results,
                    "profiling": profiling_results,
                    "timestamp": datetime.now().isoformat(),
                }

                # Log results
                target_met = (
                    "✅ MEETS TARGET"
                    if benchmark_results[sku_key]["meets_target"]
                    else "❌ EXCEEDS TARGET"
                )
                logger.info(
                    f"\n{sku_key}: {overall_time:.2f}s (target: {self.targets.get(sku_key, 'N/A')}s) - {target_met}"
                )

            except Exception as e:
                logger.error(f"Benchmark failed for {num_skus} SKUs: {e}")
                benchmark_results[f"{num_skus//1000}k_skus"] = {
                    "num_skus": num_skus,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

        # Generate summary report
        summary = self._generate_summary_report(benchmark_results)

        # Save results
        self._save_results(benchmark_results, summary)

        return {"benchmark_results": benchmark_results, "summary": summary}

    def _convert_to_transaction_matrix(
        self, transactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert transaction data to binary matrix format."""
        # Create binary transaction matrix
        matrix = transactions_df.pivot_table(
            index="transaction_id",
            columns="product_id",
            values="quantity",
            fill_value=0,
            aggfunc="sum",
        )

        # Convert to binary (0/1)
        matrix = (matrix > 0).astype(int)

        return matrix

    def _extract_hotspots(self, stats: pstats.Stats) -> List[Dict[str, Any]]:
        """Extract performance hotspots from profiling statistics."""
        # Get function statistics
        function_stats = []

        for func_key, (
            call_count,
            total_time,
            cumulative_time,
            callers,
        ) in stats.stats.items():
            filename, line_number, function_name = func_key

            # Filter for our modules only
            if any(
                module in filename
                for module in [
                    "layout_optimizer",
                    "association_engine",
                    "optimization_heuristics",
                ]
            ):
                function_stats.append(
                    {
                        "function": function_name,
                        "filename": filename.split("/")[-1],  # Just filename
                        "line_number": line_number,
                        "call_count": call_count,
                        "total_time": total_time,
                        "cumulative_time": cumulative_time,
                        "time_per_call": (
                            total_time / call_count if call_count > 0 else 0
                        ),
                    }
                )

        # Sort by cumulative time (biggest hotspots first)
        function_stats.sort(key=lambda x: x["cumulative_time"], reverse=True)

        return function_stats[:10]  # Top 10 hotspots

    def _generate_summary_report(
        self, benchmark_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary report of benchmark results."""
        summary = {
            "overall_performance": {},
            "target_compliance": {},
            "performance_trends": {},
            "recommendations": [],
        }

        # Overall performance metrics
        total_times = []
        sku_counts = []

        for sku_key, results in benchmark_results.items():
            if "error" not in results:
                total_times.append(results["overall_time"])
                sku_counts.append(results["num_skus"])

        if total_times:
            summary["overall_performance"] = {
                "fastest_run": min(total_times),
                "slowest_run": max(total_times),
                "average_time": np.mean(total_times),
                "total_skus_tested": sum(sku_counts),
            }

        # Target compliance
        compliant_runs = 0
        total_runs = 0

        for sku_key, results in benchmark_results.items():
            if "error" not in results and "meets_target" in results:
                total_runs += 1
                if results["meets_target"]:
                    compliant_runs += 1

        summary["target_compliance"] = {
            "compliant_runs": compliant_runs,
            "total_runs": total_runs,
            "compliance_rate": compliant_runs / total_runs if total_runs > 0 else 0,
        }

        # Performance analysis and recommendations
        if (
            "10k_skus" in benchmark_results
            and "error" not in benchmark_results["10k_skus"]
        ):
            result_10k = benchmark_results["10k_skus"]

            if result_10k["meets_target"]:
                summary["recommendations"].append(
                    "✅ 10k SKU target met - performance is acceptable"
                )
            else:
                summary["recommendations"].append(
                    "❌ 10k SKU target exceeded - optimization needed"
                )

                # Analyze hotspots for recommendations
                if "profiling" in result_10k and "hotspots" in result_10k["profiling"]:
                    hotspots = result_10k["profiling"]["hotspots"]
                    if hotspots:
                        top_hotspot = hotspots[0]
                        summary["recommendations"].append(
                            f"🔥 Primary hotspot: {top_hotspot['function']} "
                            f"({top_hotspot['cumulative_time']:.2f}s cumulative)"
                        )

        return summary

    def _save_results(self, benchmark_results: Dict[str, Any], summary: Dict[str, Any]):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results as JSON
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(benchmark_results, f, indent=2, default=str)

        # Save summary as JSON
        summary_file = self.output_dir / f"benchmark_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Generate visualization
        self._create_performance_charts(benchmark_results, timestamp)

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Summary saved to {summary_file}")

    def _create_performance_charts(
        self, benchmark_results: Dict[str, Any], timestamp: str
    ):
        """Create performance visualization charts."""
        try:
            # Extract data for plotting
            sku_counts = []
            overall_times = []
            target_times = []

            for sku_key, results in benchmark_results.items():
                if "error" not in results:
                    sku_counts.append(results["num_skus"])
                    overall_times.append(results["overall_time"])
                    target_times.append(results["target_time"])

            if not sku_counts:
                return

            # Create performance chart
            plt.figure(figsize=(12, 8))

            # Main performance plot
            plt.subplot(2, 2, 1)
            plt.plot(
                sku_counts,
                overall_times,
                "bo-",
                label="Actual Time",
                linewidth=2,
                markersize=8,
            )
            plt.plot(sku_counts, target_times, "r--", label="Target Time", linewidth=2)
            plt.xlabel("Number of SKUs")
            plt.ylabel("Time (seconds)")
            plt.title("Optimizer Performance vs SKU Count")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Performance breakdown
            plt.subplot(2, 2, 2)
            components = [
                "initialization",
                "association_engine",
                "layout_optimization",
                "heuristic_scoring",
            ]
            colors = ["skyblue", "lightgreen", "lightcoral", "lightyellow"]

            for i, sku_key in enumerate(["1k_skus", "5k_skus", "10k_skus"]):
                if (
                    sku_key in benchmark_results
                    and "error" not in benchmark_results[sku_key]
                ):
                    result = benchmark_results[sku_key]
                    times = [
                        result["initialization"]["initialization_time"],
                        result["association_engine"]["training_time"],
                        result["layout_optimization"]["layout_optimization_time"],
                        result["heuristic_scoring"]["heuristic_scoring_time"],
                    ]

                    bottom = 0
                    for j, (component, time_val) in enumerate(zip(components, times)):
                        plt.bar(
                            i,
                            time_val,
                            bottom=bottom,
                            color=colors[j],
                            label=component if i == 0 else "",
                        )
                        bottom += time_val

            plt.xlabel("SKU Count")
            plt.ylabel("Time (seconds)")
            plt.title("Performance Breakdown by Component")
            plt.xticks(range(3), ["1k", "5k", "10k"])
            plt.legend()

            # Memory usage
            plt.subplot(2, 2, 3)
            memory_usage = []
            for sku_key in ["1k_skus", "5k_skus", "10k_skus"]:
                if (
                    sku_key in benchmark_results
                    and "error" not in benchmark_results[sku_key]
                ):
                    memory_usage.append(
                        benchmark_results[sku_key]["initialization"]["memory_usage_mb"]
                    )
                else:
                    memory_usage.append(0)

            plt.bar(range(len(memory_usage)), memory_usage, color="lightblue")
            plt.xlabel("SKU Count")
            plt.ylabel("Memory Usage (MB)")
            plt.title("Memory Usage by SKU Count")
            plt.xticks(range(len(memory_usage)), ["1k", "5k", "10k"])

            # Target compliance
            plt.subplot(2, 2, 4)
            compliance_data = []
            labels = []

            for sku_key, results in benchmark_results.items():
                if "error" not in results and "meets_target" in results:
                    labels.append(sku_key.replace("_skus", ""))
                    compliance_data.append(1 if results["meets_target"] else 0)

            colors_compliance = ["green" if x == 1 else "red" for x in compliance_data]
            plt.bar(
                range(len(compliance_data)), compliance_data, color=colors_compliance
            )
            plt.xlabel("SKU Count")
            plt.ylabel("Meets Target (1=Yes, 0=No)")
            plt.title("Target Compliance")
            plt.xticks(range(len(labels)), labels)
            plt.ylim(-0.1, 1.1)

            plt.tight_layout()

            # Save chart
            chart_file = self.output_dir / f"performance_chart_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Performance chart saved to {chart_file}")

        except Exception as e:
            logger.error(f"Failed to create performance charts: {e}")


def main():
    """Run the benchmark suite."""
    print("🚀 Starting Optimizer Performance Benchmark Suite")
    print("=" * 60)

    # Create benchmark instance
    benchmark = OptimizerBenchmark()

    # Run full benchmark suite
    results = benchmark.run_full_benchmark_suite()

    # Print summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)

    summary = results["summary"]

    if "overall_performance" in summary:
        perf = summary["overall_performance"]
        print(f"⏱️  Fastest run: {perf['fastest_run']:.2f}s")
        print(f"⏱️  Slowest run: {perf['slowest_run']:.2f}s")
        print(f"⏱️  Average time: {perf['average_time']:.2f}s")

    if "target_compliance" in summary:
        compliance = summary["target_compliance"]
        rate = compliance["compliance_rate"] * 100
        print(
            f"🎯 Target compliance: {compliance['compliant_runs']}/{compliance['total_runs']} ({rate:.1f}%)"
        )

    if "recommendations" in summary:
        print("\n📋 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"   {rec}")

    print(f"\n📁 Detailed results saved to: {benchmark.output_dir}")
    print("✅ Benchmark complete!")


if __name__ == "__main__":
    main()
