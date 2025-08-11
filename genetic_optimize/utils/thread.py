from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Callable, List, Any, Optional

class ParallelExecutor:
    def __init__(self, num_workers: int=None):
        self.num_workers = num_workers

    def execute(self, 
                tasks: List[Any], 
                task_fn: Callable, 
                post_process: Optional[Callable] = None,
                desc: str = None) -> List[Any]:
        """
        Generic parallel execution function.

        :param tasks: List of tasks, each element is passed to `task_fn` for processing
        :param task_fn: Task execution function, each task will execute this function in parallel
        :param post_process: (Optional) Post-processing function, receives `future.result()` as parameter
        :param desc: Progress bar description
        :return: List of task execution results
        """
        results = []
        futures = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks
            futures = [executor.submit(task_fn, task) for task in tasks]

            # Progress bar
            if desc:
                progress_bar = tqdm(total=len(futures), desc=desc,
                                    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}")

            # Process results
            for future in as_completed(futures):
                result = future.result()
                if post_process:
                    result = post_process(result)
                    
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)

                if desc:
                    progress_bar.update(1)

            if desc:
                progress_bar.close()
        
        return results
    
if __name__ == "__main__":
    # Usage example
    def process_task(self, task):
        i, j = task
        print(f"{i}, {j}")
        return (i - 1, j - 1)

    def post_task(result):
        i, j = result
        return (i + 1, j + 1)
    
    executor = ParallelExecutor(num_workers=10)
    tasks = [(i, i + 1) for i in enumerate(10)]
    results = executor.execute(
        tasks=tasks, 
        task_fn=process_task,  # Directly pass the method to be called
        post_process=post_task, # No need to provide post_process if data doesn't need processing
        desc="Processing tasks"
    )
