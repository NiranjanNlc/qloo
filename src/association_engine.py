from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from itertools import combinations
from collections import defaultdict, Counter
import warnings


class AssociationEngine(ABC):
    """
    Abstract Base Class (Interface) for an association rule mining engine.

    This class defines the essential methods that any association engine
    implementation must have. This ensures a consistent API for training
    the engine and retrieving association rules, regardless of the
    underlying algorithm (e.g., Apriori, FP-Growth).
    """

    @abstractmethod
    def train(self, transactions_df: pd.DataFrame):
        """
        Trains the association rule mining algorithm on a dataset of transactions.

        Args:
            transactions_df (pd.DataFrame): A DataFrame where each row represents
                                          a transaction and columns represent items
                                          in that transaction.
        """
        pass

    @abstractmethod
    def get_associations(self, product_id: int, top_k: int = 5) -> list:
        """
        Gets the top K associated products for a given product ID.

        Args:
            product_id (int): The ID of the product to find associations for.
            top_k (int): The number of top associated products to return.

        Returns:
            list: A list of tuples or objects, each representing an associated
                  product and the strength of the association (e.g., confidence or lift).
                  Example: [(associated_product_id, score), ...]
        """
        pass


class AprioriAssociationEngine(AssociationEngine):
    """
    Concrete implementation of AssociationEngine using the Apriori algorithm.

    This implementation finds frequent itemsets and generates association rules
    based on minimum support and confidence thresholds.
    """

    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5,
        min_lift: float = 1.0,
        max_itemset_size: int = 3,
    ):
        """
        Initialize the Apriori Association Engine.

        Args:
            min_support: Minimum support threshold for frequent itemsets (default: 0.1)
            min_confidence: Minimum confidence threshold for association rules (default: 0.5)
            min_lift: Minimum lift threshold for association rules (default: 1.0)
            max_itemset_size: Maximum size of itemsets to consider (default: 3)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_itemset_size = max_itemset_size

        # Will be populated during training
        self.transactions = []
        self.frequent_itemsets = {}
        self.association_rules = []
        self.item_support = {}
        self.is_trained = False

    def train(self, transactions_df: pd.DataFrame):
        """
        Train the association engine on transaction data using Apriori algorithm.

        Args:
            transactions_df: DataFrame with transactions. Expected format:
                            - transaction_id column
                            - product_id column
                            or multiple columns with product IDs (one-hot encoded)
        """
        print("Training association engine...")

        # Convert DataFrame to list of transactions (sets of items)
        self.transactions = self._prepare_transactions(transactions_df)

        if not self.transactions:
            warnings.warn("No transactions found. Please check your data format.")
            return

        # Step 1: Find frequent itemsets using Apriori
        self.frequent_itemsets = self._find_frequent_itemsets()

        # Step 2: Generate association rules
        self.association_rules = self._generate_association_rules()

        # Step 3: Calculate item support for quick lookup
        self._calculate_item_support()

        self.is_trained = True
        print(
            f"Training complete! Found {len(self.association_rules)} association rules."
        )

    def _prepare_transactions(self, df: pd.DataFrame) -> List[Set[int]]:
        """Convert DataFrame to list of transaction sets."""
        transactions = []

        if "transaction_id" in df.columns and "product_id" in df.columns:
            # Format: transaction_id, product_id
            grouped = df.groupby("transaction_id")["product_id"].apply(set).tolist()
            transactions = grouped
        else:
            # Assume one-hot encoded format or basket format
            for _, row in df.iterrows():
                # Get non-zero/non-null product IDs
                transaction = set()
                for col, val in row.items():
                    if pd.notna(val) and val != 0 and val != "":
                        try:
                            transaction.add(int(col) if col.isdigit() else int(val))
                        except (ValueError, TypeError):
                            continue
                if transaction:
                    transactions.append(transaction)

        return transactions

    def _find_frequent_itemsets(self) -> Dict[int, Dict[frozenset, int]]:
        """Find frequent itemsets using Apriori algorithm."""
        frequent_itemsets = {}
        total_transactions = len(self.transactions)
        min_support_count = int(self.min_support * total_transactions)

        # Start with frequent 1-itemsets
        item_counts = Counter()
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1

        # Keep only frequent 1-itemsets
        frequent_1_itemsets = {
            frozenset([item]): count
            for item, count in item_counts.items()
            if count >= min_support_count
        }

        if not frequent_1_itemsets:
            return {}

        frequent_itemsets[1] = frequent_1_itemsets

        # Generate larger itemsets
        k = 2
        while k <= self.max_itemset_size:
            candidates = self._generate_candidates(frequent_itemsets[k - 1])
            if not candidates:
                break

            # Count support for candidates
            candidate_counts = defaultdict(int)
            for transaction in self.transactions:
                for candidate in candidates:
                    if candidate.issubset(transaction):
                        candidate_counts[candidate] += 1

            # Keep only frequent candidates
            frequent_k_itemsets = {
                itemset: count
                for itemset, count in candidate_counts.items()
                if count >= min_support_count
            }

            if not frequent_k_itemsets:
                break

            frequent_itemsets[k] = frequent_k_itemsets
            k += 1

        return frequent_itemsets

    def _generate_candidates(
        self, frequent_itemsets: Dict[frozenset, int]
    ) -> List[frozenset]:
        """Generate candidate itemsets from frequent itemsets of size k-1."""
        candidates = []
        itemsets = list(frequent_itemsets.keys())

        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                # Join two itemsets if they differ by exactly one item
                union = itemsets[i] | itemsets[j]
                if len(union) == len(itemsets[i]) + 1:
                    # Check if all subsets of size k-1 are frequent
                    if self._all_subsets_frequent(union, frequent_itemsets):
                        candidates.append(union)

        return candidates

    def _all_subsets_frequent(
        self, candidate: frozenset, frequent_itemsets: Dict[frozenset, int]
    ) -> bool:
        """Check if all subsets of candidate are frequent."""
        for item in candidate:
            subset = candidate - {item}
            if subset not in frequent_itemsets:
                return False
        return True

    def _generate_association_rules(self) -> List[Dict]:
        """Generate association rules from frequent itemsets."""
        rules = []

        # Generate rules from itemsets of size 2 or larger
        for size in range(2, len(self.frequent_itemsets) + 1):
            if size not in self.frequent_itemsets:
                continue

            for itemset, support_count in self.frequent_itemsets[size].items():
                # Generate all possible antecedent -> consequent rules
                for r in range(1, len(itemset)):
                    for antecedent in combinations(itemset, r):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent

                        # Calculate confidence
                        antecedent_support = self._get_support_count(antecedent)
                        if antecedent_support > 0:
                            confidence = support_count / antecedent_support

                            if confidence >= self.min_confidence:
                                # Calculate lift
                                consequent_support = self._get_support_count(consequent)
                                total_transactions = len(self.transactions)

                                if consequent_support > 0:
                                    expected_confidence = (
                                        consequent_support / total_transactions
                                    )
                                    lift = confidence / expected_confidence

                                    if lift >= self.min_lift:
                                        rules.append(
                                            {
                                                "antecedent": antecedent,
                                                "consequent": consequent,
                                                "support": support_count
                                                / total_transactions,
                                                "confidence": confidence,
                                                "lift": lift,
                                                "conviction": self._calculate_conviction(
                                                    confidence, expected_confidence
                                                ),
                                            }
                                        )

        # Sort rules by lift (descending)
        rules.sort(key=lambda x: x["lift"], reverse=True)
        return rules

    def _get_support_count(self, itemset: frozenset) -> int:
        """Get support count for an itemset."""
        itemset_size = len(itemset)
        if itemset_size in self.frequent_itemsets:
            return self.frequent_itemsets[itemset_size].get(itemset, 0)
        return 0

    def _calculate_conviction(
        self, confidence: float, expected_confidence: float
    ) -> float:
        """Calculate conviction metric."""
        if confidence == 1.0:
            return float("inf")
        return (1 - expected_confidence) / (1 - confidence)

    def _calculate_item_support(self):
        """Calculate support for individual items."""
        if 1 in self.frequent_itemsets:
            total_transactions = len(self.transactions)
            self.item_support = {
                list(itemset)[0]: count / total_transactions
                for itemset, count in self.frequent_itemsets[1].items()
            }

    def get_associations(
        self, product_id: int, top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Get top K associated products for a given product ID.

        Args:
            product_id: The ID of the product to find associations for
            top_k: Number of top associations to return

        Returns:
            List of tuples (associated_product_id, association_strength)
        """
        if not self.is_trained:
            warnings.warn("Engine not trained yet. Call train() first.")
            return []

        associations = []

        # Find rules where product_id is in the antecedent
        for rule in self.association_rules:
            if product_id in rule["antecedent"]:
                for consequent_item in rule["consequent"]:
                    associations.append((consequent_item, rule["lift"]))
            elif product_id in rule["consequent"]:
                # Also consider reverse associations
                for antecedent_item in rule["antecedent"]:
                    associations.append((antecedent_item, rule["lift"]))

        # Remove duplicates and sort by association strength
        unique_associations = {}
        for item, strength in associations:
            if item not in unique_associations or strength > unique_associations[item]:
                unique_associations[item] = strength

        # Sort by strength and return top_k
        sorted_associations = sorted(
            unique_associations.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_associations[:top_k]

    def get_rule_details(self, product_id: int) -> List[Dict]:
        """Get detailed rule information for a product."""
        if not self.is_trained:
            return []

        relevant_rules = []
        for rule in self.association_rules:
            if product_id in rule["antecedent"] or product_id in rule["consequent"]:
                relevant_rules.append(rule)

        return relevant_rules

    def get_stats(self) -> Dict:
        """Get training statistics."""
        if not self.is_trained:
            return {"status": "Not trained"}

        return {
            "total_transactions": len(self.transactions),
            "total_rules": len(self.association_rules),
            "frequent_itemsets_by_size": {
                k: len(v) for k, v in self.frequent_itemsets.items()
            },
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "min_lift": self.min_lift,
            "unique_items": len(self.item_support) if self.item_support else 0,
        }
