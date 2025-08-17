from multiprocessing import Pool, Manager
from typing import List, Dict
from tqdm import tqdm
import time
from preverification.config import (
    model_api_name,
    max_thread_num,
    silicon_api_key
)
from openai import OpenAI
import logging
def get_default_client_sync():
    return OpenAI(
            api_key=silicon_api_key,
            base_url="https://api.siliconflow.cn/v1",
        )

def worker_conservation(args):
    conversations, counter, total, lock = args
    client = get_default_client_sync()
    results = []
    for i, (conversation, need_json) in enumerate(conversations):
        while True:
            try:
                result = client.chat.completions.create(
                            model=model_api_name,
                            messages=conversation,
                            stream=False,
                            temperature=01.7,
                )
                results.append(result.choices[0].message.content)
                break
            except Exception as e:
                print(f"Error occurred: {e}. Retrying...")
                time.sleep(60)
                continue
        if counter is not None:
            with lock:
                counter.value += 1

    return results


def multi_process_api_conservation(
    conversations: List[List[Dict[str, str]]],
    num_processes: int = 8,
    show_progress: bool = False,
) -> List[str]:
    chunk_size = max(len(conversations) // num_processes, 1)
    conversations_chunks = [
        conversations[i : i + chunk_size]
        for i in range(0, len(conversations), chunk_size)
    ]

    manager = Manager()
    counter = manager.Value("i", 0) if show_progress else None
    lock = manager.Lock() if show_progress else None
    total = len(conversations)

    pool = Pool(processes=num_processes)

    # 把计数器传给worker
    args = [(chunk, counter, total, lock) for chunk in conversations_chunks]

    results_async = [pool.apply_async(worker_conservation, (arg,)) for arg in args]
    import time

    if show_progress:
        with tqdm(total=total) as pbar:
            last_val = 0
            while True:
                current_val = counter.value
                delta = current_val - last_val
                if delta > 0:
                    pbar.update(delta)
                    last_val = current_val
                if current_val >= total:
                    break
                time.sleep(2)

    pool.close()
    pool.join()

    final_results = []
    for r in results_async:
        final_results.extend(r.get())

    return final_results


def multi_conservation(
    system_prompt: List[str],
    user_input: List[str],
    need_json: False,
    show_progress: bool = False,
) -> List[str]:
    """多并行单轮对话请求
    :param system_prompt: 系统提示词, List[str]
    :param user_input: 用户输入, List[str],与system_prompt一一对应
    :param need_json: 是否需要json格式的返回, List[bool] or bool
    :param show_progress: 是否显示进度条, bool
    """
    if isinstance(need_json, bool):
        need_json = [need_json] * len(system_prompt)
    conversations = []
    for prompt, user, nj in zip(system_prompt, user_input, need_json):
        conversations.append(
            (
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user},
                ],
                nj,
            )
        )
    results = multi_process_api_conservation(
        conversations, num_processes=max_thread_num, show_progress=show_progress
    )
    logging.info(f"{len(results)} Task Finished!")
    return results


def single_conversation(
    system_prompt: str,
    user_input: str,
    need_json: bool = False,
    show_progress: bool = False,
) -> str:
    """
    单次对话请求
    :param system_prompt: 系统提示词,str
    :param user_input: 用户输入,str
    :param need_json: 是否需要json格式的返回, bool
    :param show_progress: 是否显示进度条, bool

    """

    conversations = [
        (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            need_json,
        )
    ]
    results = multi_process_api_conservation(
        conversations, num_processes=max_thread_num, show_progress=show_progress
    )
    return results[0] if results else ""

if __name__=="__main__":
    from multiprocessing import freeze_support
    freeze_support() # 多线程必须使用！！
    user_input = ["Hello"]*5
    system_prompt = ["You are a helpful assistant."]*5
    need_json=[False]*5
    results = multi_conservation(system_prompt, user_input, need_json,show_progress=True)
    print(results)