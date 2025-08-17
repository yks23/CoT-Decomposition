## Preliminary Experimental Results

### 1. Zero-shot Prompt vs. No Prompt

| Dataset | Model                 | With Prompt (Zero-shot) | Without Prompt | Observation                                |
| ------- | --------------------- | ----------------------- | -------------- | ------------------------------------------ |
| GSM8K   | LLaMA 3.1 8B Instruct | 0.7809                  | 0.8446         | Prompt hurts performance due to OOD shift. |
| GSM8K   | Mistral 7B Instruct   | 0.5572                  | 0.6111         | Same as above.                             |
| MATH    | LLaMA 3.1 8B Instruct | 0.6100                  | 0.6370         | Same as above.                             |
| MATH    | Mistral 7B Instruct   | 0.2750                  | 0.2805         | Same as above.                             |

**Reason:**
During training, the model learns $p(x \mid \text{question})$. Adding a prompt $c$ changes the distribution to $p(x \mid \text{question}, c)$, which differs from the training distribution.

---

### 2. Prompt with Temperature Scheduling
For this experiment, we used the LLaMA 3.1 8B model with a zero-shot prompt:
```
For each problem:  
1. "PLAN": briefly plan your solution strategy. Think about the steps you need to solve the problem.
2. "ROLL OUT": solve the problem step by step according to your plan.
3. "ANSWER": provide a concise answer as a number.
```
and we use temp1 for PLAN, temp2 for ROLL OUT and ANSWER.

#### LLaMA 3.1 8B — With Prompt Results

| Dataset | Temp Setting | @1     | @2     | @3     | @4     | @5     |  @6    |  @7    |
| ------- | ------------ | ------ | ------ | ------ | ------ | ------ |------ | ------ |
| GSM8K   | Temp = 0     | —      | —      | —      | —      | —      |—      | 86.66% |

| GSM8K   | 1.0 + 1.0    | 80.74% | 88.86% |91.74% | 93.48% | 94.09% |  94.77 | 95.07 | 95.68
| GSM8K   | 1.3 + 0.4    |  83.93% | 91.36% | 93.71% | 94.62% | 95.53% | 96.74 | 97.10 | 97.37
| GSM8K   | 1.0 + 0.0    | 82.08% | 89.92% | 92.49% | 93.71% | 94.39% | 95.23 | 95.68 | 96.09
| GSM8K   | 1.0 + 0.4    | 85.23% | 90.67% | 93.40% | 94.84% | 95.53% | 96.51 | 96.74 | 97.01

| Dataset | Temp Setting | @1     | @2     | @3     | @4     | @5     |
| ------- | ------------ | ------ | ------ | ------ | ------ | ------ |
| MATH    | Temp = 0     | —      | —      | —      | —      | 62.43%      | 
| MATH    | Temp = 1     | 61.50%      | 72.28%      | 77.84%      | 79.78%    |  80.54% | 
| MATH    | 1.0  + 0.0   | 63.30%      |  68.41%      | 69.84%      | 71.19%      | 72.70%      | 

MATH 1.3 + 0.4 onprocess
MATH 1.0 + 0.4 onprocess



（MATH/algebra is still being calculated）



---

### 3. Key Findings

1. Low temperature is beneficial for problem solving.
2. For multi-sample generation, introducing some stochasticity is necessary.
3. We hypothesize that final accuracy can be modeled as:

   $$
   \text{Accuracy} \approx f(\text{deterministic}) + \frac{1}{1 + e^{-k}} \cdot g(\text{stochastic}), \quad k = \#\text{samples}
   $$
4. Generally, deterministic and stochastic components are **constructive** when combined.
   If we can separate them, there is room for more targeted improvements.
5. Our method to decouple the two components seems novel and interesting. However, current prompt engineering is not sufficient.

---

### Next Stage

1. **SFT**: Perform supervised fine-tuning (SFT) on *deepseek-7b-math-instruct* so it can output in our desired format.Our code base is verl.
2. **RL**: Use entropy-related rewards to make the stochastic part shorter and more diverse, and the deterministic part more precise.

---

If it is convenient, could Rushi share your experiment results with us?

---


      100   200   300
test  0.778 0.838 0.82
train 0.85  0.93 0.97
trainset ~ 7k
200 epochs * 1e-5 * cos* 144