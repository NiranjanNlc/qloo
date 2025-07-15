from abc import ABC, abstractmethod
import pandas as pd

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