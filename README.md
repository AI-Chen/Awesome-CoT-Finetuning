# Awesome-CoT-Finetuning

A collection of paper and code for the chain of thought finetuning (CoT-Finetuning). We are looking forward to other participants to share their papers and codes. If interested, please contact chenxs@nudt.edu.cn or xschenranker@gmail.com. :fire: :fire: :fire: 

- We reproduced the code of all collected papers and tried to compare them under the same training framework. The reproduced code can be viewed in the [code](./code). :fire: :fire: :fire:
- We will release all datasets we construct in the same way as described in all collected papers in the future.

:bell: :bell: :bell: Update at July 2024


# Bookmarks
- [Survey Papers](#survey-papers-)
- [Datasets](#datasets-)
- [Chain-of-Thought Distillation](#Chain-of-Thought-Distillation-)
- [Self-Enhancement](#Self-Enhancement-)
- [Application](#Applicationt-)

## Survey Papers <span id="survey-papers-"></span>
| **Year**   | **Title**                                                                                     |  **Venue**    |                                       **Paper**                                            | **Code** | 
| ---- |----------------------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------:|:----:|
| 2024  | **A Survey on Knowledge Distillation of Large Language Models**   |  -    |                   [Link](https://arxiv.org/pdf/2402.13116)                    | [Link](https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs)  | 
| 2024  | **Navigate through Enigmatic Labyrinth A Survey of Chain of Thought Reasoning: Advances, Frontiers and Future**   |  ACL    |                   [Link](https://arxiv.org/abs/2309.15402)                    | -  | 

## Datasets <span id="datasets-"></span>
| **Type**                                                                                     |  **Dataset**    |                                       **Description**                                            | **Download** |**Samples**|**Choices**|**Manual Rationale**|
|----------------------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| Commonsense   |  CommonsenseQA    |                   [Link](https://arxiv.org/abs/1811.00937)                    | [Link](https://www.tau-nlp.org/commonsenseqa)  | 12,102 | 5 | No |
| Commonsense   |   StrategyQA   |                   [Link](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00370/100680/Did-Aristotle-Use-a-Laptop-A-Question-Answering)                    | [Link](https://allenai.org/data/strategyqa)  | 2780 | 2 | Yes |
| Commonsense   |   OpenBookQA   |                   [Link](https://aclanthology.org/D18-1260.pdf)                    | [Link](https://leaderboard.allenai.org/open_book_qa/submissions/get-started)  | 5,957 | 4 | No |
| Arithmetic   |   SingleEq   |                   [Link](https://aclanthology.org/Q15-1042.pdf)                    | [Link](https://gitlab.cs.washington.edu/ALGES/TACL2015)  | 508| - | Yes |
| Arithmetic   |   AddSub   |                   [Link](https://aclanthology.org/D14-1058.pdf)                    | [Link](https://www.cs.washington.edu/nlp/arithmetic)  |395 | - | Yes |
| Arithmetic   |   MultiArith   |                   [Link](https://aclanthology.org/D15-1202.pdf)                    | [Link](https://github.com/wangxr14/Algebraic-Word-Problem-Solver/tree/master/data)  | 600|- | Yes |
| Arithmetic   |   GSM8K   |                   [Link](https://arxiv.org/pdf/2110.14168)                    | [Link](https://github.com/openai/grade-school-math/tree/master/grade_school_math/data)  | 8000| - | Yes |
| Arithmetic   |   SVAMP   |                   [Link](https://aclanthology.org/2021.naacl-main.168.pdf)                    | [Link](https://github.com/arkilpatel/SVAMP)  |1000 | - | Yes|
| Arithmetic   |   AQUA   |                   [Link](https://aclanthology.org/P17-1015.pdf)                    | [Link](https://github.com/google-deepmind/AQuA)  |100,000 | 5 | Yes |
| Arithmetic   |   ASDiv   |                   [Link](https://aclanthology.org/2020.acl-main.92.pdf)                    | [Link](https://github.com/chaochun/nlu-asdiv-dataset/blob/master/dataset/ASDiv.xml)  | 2305| -| Yes|
| Arithmetic   |   MATH   |                   [Link](https://arxiv.org/pdf/2103.03874)                    | [Link](https://github.com/hendrycks/math)  |12,500 | - | Yes|
| Symbolic   |  Last Letter Concatenation    |                   [Link](https://arxiv.org/pdf/2201.11903)                    | [Link](https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/dataset/last_letters)  |500 | - | No |
| Symbolic   |  Coin Flip    |                   [Link](https://arxiv.org/pdf/2201.11903)                    | [Link](https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/dataset/coin_flip)  |500 | 2 | No |
| Science   |  SciQ    |                   [Link](https://aclanthology.org/W17-4413.pdf)                    | [Link](https://allenai.org/data/sciq)  |13,679  | 4 | No|
| Science   |   ARC-Easy  |                   [Link](https://arxiv.org/pdf/1803.05457)                    | [Link](https://allenai.org/data/arc)  | 5197 | 4 | No |
| Science   |   ARC-Challenge   |                   [Link](https://arxiv.org/pdf/1803.05457)                    | [Link](https://allenai.org/data/arc)  | 2590 | 4 | Yes |
|  Natural Language Inference   |  e-SNLI    |                   [Link](https://papers.nips.cc/paper_files/paper/2018/file/4c7a167bb329bd92580a99ce422d6fa6-Paper.pdf)                    | [Link](https://github.com/OanaMariaCamburu/e-SNLI/tree/master/dataset)  |569,033 | 3 | Yes |
|  Natural Language Inference   |  ANLI-R1    |                   [Link](https://aclanthology.org/2020.acl-main.441.pdf)                    | [Link](https://github.com/facebookresearch/anli)  |18,946 | 3 | No | 
| Generic ability   |  Big Bench Hard    |                   [Link](https://arxiv.org/pdf/2206.04615)                    | [Link](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks)  |24 tasks | \ | No|


## Chain-of-Thought Distillation <span id="Chain-of-Thought-Distillation-"></span>
| **Year**   | **Title**                                                                                     |  **Venue**    |                                       **Paper**                                            | **Code** |
| ---- |----------------------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------:|:----:|
| 2022  | **Large Language Models Are Reasoning Teachers**   |  ACL    |                   [Link](https://arxiv.org/pdf/2212.10071)                    | [Link](https://github.com/itsnamgyu/reasoning-teacher)   |
| 2023  | **Teaching Small Language Models to Reason**   |  ACL    |                   [Link](https://aclanthology.org/2023.acl-short.151.pdf)                    | -   |
| 2023  | **Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes**   |  ACL    |                   [Link](https://aclanthology.org/2023.findings-acl.507.pdf)                    | [Link](https://github.com/Yiwei98/TDG)   |
| 2023  | **SCOTT: Self-Consistent Chain-of-Thought Distillation**   |  ACL    |                   [Link](https://aclanthology.org/2023.acl-long.304v2.pdf)                    | [Link](https://github.com/wangpf3/consistent-CoT-distillation)   |
| 2023  | **Distilling Reasoning Capabilities into Smaller Language Models**   |  ACL    |                   [Link](https://aclanthology.org/2023.findings-acl.441.pdf)                    | [Link](https://github.com/kumar-shridhar/Distiiling-LM)   |
| 2023  | **Orca: Progressive Learning from Complex Explanation Traces of GPT-4**   |  -    |                   [Link](https://arxiv.org/pdf/2306.02707)                    | -   |
| 2023  | **Specializing Smaller Language Models towards Multi-Step Reasoning**   |  PMLR    |                   [Link](https://proceedings.mlr.press/v202/fu23d/fu23d.pdf)                    | [Link](https://github.com/FranxYao/FlanT5-CoT-Specialization)   |
| 2023  | **Symbolic Chain-of-Thought Distillation: Small Models Can Also “Think” Step-by-Step**   |  ACL    |                   [Link](https://aclanthology.org/2023.acl-long.150.pdf)                    | -   |
| 2023  | **IMPLICIT CHAIN OF THOUGHT REASONING VIA KNOWLEDGE DISTILLATION**   |  -    |                   [Link](https://arxiv.org/pdf/2311.01460)                    | [Link](https://github.com/da03/implicit_chain_of_thought)   |
| 2024  | **Turning Dust into Gold: Distilling Complex Reasoning Capabilities from LLMs by Leveraging Negative Data**   |  AAAI    |                   [Link](https://ojs.aaai.org/index.php/AAAI/article/view/29821)                    | [Link](https://github.com/Yiwei98/TDG)   |
| 2024  | **Enhancing Code Generation Performance of Smaller Models by Distilling the Reasoning Ability of LLMs**   |  COLING    |                   [Link](https://aclanthology.org/2024.lrec-main.521.pdf)                    | [Link](https://github.com/sssszh/CodePLAN)   |
| 2024  | **PaD: Program-aided Distillation Can Teach Small Models Reasoning Better than Chain-of-thought Fine-tuning**   |  NAACL    |                   [Link](https://aclanthology.org/2024.naacl-long.142.pdf)                    | [Link](https://github.com/Xuekai-Zhu/pad)   |
| 2024  | **Mind’s Mirror: Distilling Self-Evaluation Capability and Comprehensive Thinking from Large Language Models**   |  NAACL    |                   [Link](https://aclanthology.org/2024.naacl-long.376.pdf)                    | [Link](https://github.com/Attention-is-All-I-Need/Mind-s-Mirror-Distilling-LLM)   |
| 2024  | **Mixed Distillation Helps Smaller Language Model Better Reasoning**   | -    |                   [Link](https://arxiv.org/pdf/2312.10730)                    | -   |
| 2024  | **Distilling Mathematical Reasoning Capabilities into Small Language Models**   | -    |                   [Link](https://arxiv.org/pdf/2401.11864)                    |  -     |

## Self-Enhancement <span id="Self-Enhancement-"></span>
| **Year**   | **Title**                                                                                     |  **Venue**    |                                       **Paper**                                            | **Code** |
| ---- |----------------------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------:|:----:|
| 2022  | **Large Language Models Can Self-Improve**   |  ACL    |                   [Link](https://aclanthology.org/2023.emnlp-main.67/)                    | [Link](https://github.com/google-research/distilling-step-by-step)  |
| 2023  | **DialCoT Meets PPO: Decomposing and Exploring Reasoning Paths in Smaller Language Models**   |  EMNLP    |                   [Link](https://aclanthology.org/2023.emnlp-main.501.pdf)                    | [Link](https://github.com/hccngu/DialCoT)  |
| 2024  | **Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning**   |  -    |                   [Link](https://arxiv.org/pdf/2405.00451)                    | [Link](https://github.com/YuxiXie/MCTS-DPO)   |
| 2024  | **Step-level Value Preference Optimization for Mathematical Reasoning**   |  -    |                   [Link](https://arxiv.org/pdf/2406.10858)                    | -  |
| 2024  | **Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations**   |  ACL    |                   [Link](https://aclanthology.org/2024.acl-long.510.pdf)                    | -  |
| 2024  | **Step-Controlled DPO: Leveraging Stepwise Error for Enhanced Mathematical Reasoning**   |  -    |                   [Link](https://arxiv.org/pdf/2407.00782)                   | [Link](https://github.com/mathllm/Step-Controlled_DPO)  |
| 2024  | **STEP-DPO: STEP-WISE PREFERENCE OPTIMIZATION FOR LONG-CHAIN REASONING OF LLMS**   |  -    |                   [Link](https://arxiv.org/pdf/2406.18629)                    | -  |

## Application <span id="Applicationt-"></span>
| **Year**   | **Title**                                                                                     |  **Venue**    |                                       **Paper**                                            | **Code** |
| ---- |----------------------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------:|:----:|
| 2024  | **Effective Distillation of Table-based Reasoning Ability from LLMs**   |  ACL    |                   [Link](https://aclanthology.org/2024.lrec-main.492/)                    | [Link]([https://github.com/Yiwei98/TDG](https://github.com/Bernard-Yang/DistillTableCoT))  |
| 2024  | **Probe then Retrieve and Reason: Distilling Probing and Reasoning Capabilities into Smaller Language Models**   |  COLING    |                   [Link](https://aclanthology.org/2024.lrec-main.1140.pdf)                    | -  |
| 2024  | **RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold**   |  -    |                   [Link](https://arxiv.org/pdf/2406.14532)                    | [Link](https://github.com/ars22/scaling-LLM-math-synthetic-data)  |
