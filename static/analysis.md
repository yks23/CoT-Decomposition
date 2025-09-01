# 2025/8/12 预实验 
# qwen2.5 1.5B - GSM8K
w/o    81.35%  :  83.70%

山不厌高:
qwen2.5 w/o prompt 没有prompt高了2个点。感觉1.gsm8k分解没啥用（较为简单）2. qwen2.5也没有按照我们的要求分解（指令遵循）3. 上下文比较长了导致有问题（horizon长度）

山不厌高:
那么我在想：
1. 把这个用来说明prompt工程没用，还是得训练
2. 换个模型/数据集再测
3. 优化提示词

# qwen2.5 7B - GSM8K
        "pred_cot": "system\nYou are a helpful and precise math solver.  \nFor each problem, you must output the solution in the following structured format:\n\n<thinking>(\n  <stochastic> A brief and intuitive idea for solving the problem (≤ 20 tokens, diverse in phrasing)\n  <deterministic> A clear and correct step-by-step explanation to reach the final answer\n)<\\thinking>\n<answer>\\boxed{Final concise answer,no latex, no math symbols, no explanation, just a number}\n\nYou need to output the special tokens <thinking> and <answer> and <stochastic> and <deterministic> in the output.\nFor each question, you give a brief and intuitive idea for solving the problem, and then give a clear and correct step-by-step explanation to reach the final answer.\nHere is a examples:\n\nQ: What is the formula for gravitational force between two objects?\n\n<thinking>( <stochastic>Calculate first download, restart, and second download times <deterministic>The file is 200GB with a 2GB/minute speed. First, Carla downloads 40% (80GB) taking 40 minutes. Then a 20-minute restart occurs. After restarting, she must download the entire 200GB again, which takes 100 minutes. Total time is 40 + 20 + 100 = 160 minutes.\n)</thinking><answer>\\boxed{160}\n
        user\nA robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?\nassistant\n
        To determine the total number of bolts of fiber needed for the robe, we need to follow these steps:\n\n1. Identify the amount of blue fiber required.\n2. Determine the amount of white fiber required.\n3. Add the amounts of blue and white fiber together.\n\nFirst, we know that the robe takes 2 bolts of blue fiber. \n\nNext, we need to find out how much white fiber is required. The problem states that the robe takes half as much white fiber as blue fiber. Therefore, we calculate the amount of white fiber as follows:\n\\[ \\text{White fiber} = \\frac{1}{2} \\times \\text{Blue fiber} = \\frac{1}{2} \\times 2 = 1 \\text{ bolt} \\]\n\nNow, we add the amounts of blue and white fiber together to find the total number of bolts:\n\\[ \\text{Total bolts} = \\text{Blue fiber} + \\text{White fiber} = 2 + 1 = 3 \\]\n\nThus, the total number of bolts needed for the robe is \\(\\boxed{3}\\).",


# llama3.1 7B - GSM8K

# mistral 7B - GSM8K



Instruct版本才可以测。mistral,llama3.1,deepseek
math版本只能微调。



