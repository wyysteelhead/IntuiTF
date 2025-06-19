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
        通用并行执行函数。

        :param tasks: 任务列表，每个元素传入 `task_fn` 处理
        :param task_fn: 任务执行函数，每个任务会并行执行该函数
        :param post_process: (可选) 后处理函数，接收 `future.result()` 作为参数
        :param desc: 进度条描述
        :return: 任务执行结果的列表
        """
        results = []
        futures = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交任务
            futures = [executor.submit(task_fn, task) for task in tasks]

            # 进度条
            if desc:
                progress_bar = tqdm(total=len(futures), desc=desc,
                                    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}")

            # 处理结果
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
    # 使用案例
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
        task_fn=process_task,  # 直接传入待调用方法
        post_process=post_task, #如果数据不需要处理就不需要给post_process
        desc="xxx"
    )
