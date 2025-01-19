import os
import time
from typing import List

import pandas as pd
import sparql_dataframe

from config import Config
from .query import Query


class MLSeaRepository:
    """
    Class to interact with the MLSea SPARQL endpoint and retrieve data.

    Args:
        sparql_endpoint (str): The URL of the SPARQL endpoint.
        use_cache (bool): Indicates whether to use cached results. Handful for development.
        cache_dir_path (str): The path to the directory where to store the cached results.
    """

    def __init__(self, sparql_endpoint: str = Config.MLSEA_SPARQL_ENDPOINT, use_cache: bool = Config.MLSEA_USE_CACHE,
                 cache_dir_path: str = Config.MLSEA_CACHE_DIR, retries: int = 3, rate_limit: int = 30):
        self._sparql_endpoint = sparql_endpoint
        self._use_cache = use_cache
        self._cache_dir_path = cache_dir_path
        self._retries = retries
        self._rate_limit = rate_limit  # maximum number of requests per minute
        self._last_request_time = 0

    def retrieve_datasets_from_openml(self, dataset_ids: List[int] = None):
        """
        Retrieves a specific dataset from OpenML or all datasets if no ID is provided.

        Args:
            dataset_ids (List[int], optional): The IDs of the datasets to retrieve.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved datasets.
        """
        if dataset_ids is None:
            return self._execute_query_with_retries(Query.RETRIEVE_ALL_DATASETS_FROM_OPENML)
        dataset_ids = " ".join([f"mlsea_openml_dataset:{dataset_id}" for dataset_id in dataset_ids])
        return self._execute_query_with_retries(Query.RETRIEVE_DATASETS_FROM_OPENML, datasetId=dataset_ids)

    def retrieve_all_tasks_from_openml_for_dataset(self, dataset_id: int):
        """
        Retrieves all tasks from OpenML for a specific dataset.

        Args:
            dataset_id (int): The ID of the dataset.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved tasks.
        """
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_TASKS_FROM_OPENML_FOR_DATASET, datasetId=dataset_id)

    def retrieve_all_evaluation_procedure_types_from_openml_for_task(self, task_id: int):
        """
        Retrieves all evaluation procedure types from OpenML for a specific task.

        Args:
            task_id (int): The ID of the task.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved evaluation procedure types.
        """
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_EVALUATION_PROCEDURE_TYPES_FROM_OPENML_FOR_TASK,
                                                taskId=task_id)

    def retrieve_all_implementations_from_openml_for_task(self, task_id: int):
        """
        Retrieves all implementations from OpenML for a specific task.

        Args:
            task_id (int): The ID of the task.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved implementations.
        """
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_IMPLEMENTATIONS_FROM_OPENML_FOR_TASK, taskId=task_id)

    def retrieve_dependencies_from_openml_for_implementation(self, implementation_id: int):
        """
        Retrieves all dependencies from OpenML for a specific implementation.

        Args:
            implementation_id (int): The ID of the implementation.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved dependencies.
        """
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_DEPENDENCIES_FROM_OPENML_FOR_IMPLEMENTATION,
                                                implementationId=implementation_id)

    def retrieve_all_runs_from_openml_for_task(self, task_id: int):
        """
        Retrieves all runs from OpenML for a specific task.

        Args:
            task_id (int): The ID of the task.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved runs.
        """
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_RUNS_FROM_OPENML_FOR_TASK, taskId=task_id)

    def retrieve_all_metrics_from_openml_for_run(self, run_id: int):
        """
        Retrieves all metrics from OpenML for a specific run.

        Args:
            run_id (int): The ID of the run.

        Returns:
            pd.DataFrame: The DataFrame with the retrieved metrics.
        """
        return self._execute_query_with_retries(Query.RETRIEVE_ALL_METRICS_FROM_OPENML_FOR_RUN, runId=run_id)

    def _execute_query(self, query: Query, **params):
        """
        Executes the given query and returns the results as a DataFrame. Uses cache if enabled.

        Args:
            query (Query): The query to execute.
            **params: Parameters for the query.

        Returns:
            pd.DataFrame: The DataFrame with the query results.
        """
        file_path = ''
        if self._use_cache:
            params_str = "_".join([str(value) for value in params.values()])
            filename = f"{query.cached_filename_prefix}_{params_str}.csv" if params else f"{query.cached_filename_prefix}.csv"
            file_path = os.path.join(self._cache_dir_path, filename)
            os.makedirs(self._cache_dir_path, exist_ok=True)
            try:
                return pd.read_csv(file_path)
            except FileNotFoundError:
                print("No cached response found for query, querying endpoint...")

        self._ensure_rate_limit()
        result_df = sparql_dataframe.get(self._sparql_endpoint, query(**params))
        if self._use_cache:
            result_df.to_csv(file_path, index=False)

        return result_df

    def _execute_query_with_retries(self, query: Query, **params):
        """
        Executes the given query with retries and returns the results as a DataFrame.

        Args:
            query (Query): The query to execute.
            **params: Parameters for the query.

        Returns:
            pd.DataFrame: The DataFrame with the query results.
        """
        for try_no in range(self._retries):
            try:
                return self._execute_query(query, **params)
            except Exception as ex:
                if try_no == self._retries - 1:
                    raise ex
                print(f"Error executing query: {ex}")
                print(f"Retrying... ({try_no + 1})")

    def _ensure_rate_limit(self):
        """
        Ensures the rate limit is not exceeded.
        """
        current_time = time.time()
        elapsed_time = current_time - self._last_request_time
        if elapsed_time < 60 / self._rate_limit:
            time.sleep(60 / self._rate_limit - elapsed_time)
        self._last_request_time = time.time()
