# Awesome-CoT-Finetuning

A collection of paper and code for the chain of thought finetuning (CoT-Finetuning). We are looking forward to other participants to share their papers and codes. If interested, please contact chenxs@nudt.edu.cn or xschenranker@gmail.com. :fire: :fire: :fire: 

:bell: :bell: :bell: Update at Oct 2025


# Bookmarks
- [Survey Papers](#survey-papers-)
- [Datasets](#datasets-)
- [Being a Thinking Model](#Thinking-)
- [Being an Insight Model](#Insight-)

## Survey Papers <span id="survey-papers-"></span>
| **Year**   | **Title**                                                                                     |  **Venue**    |                                       **Paper**                                            | **Code** | 
| ---- |----------------------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------:|:----:|
| 2024  | **A Survey on Knowledge Distillation of Large Language Models**   |  -    |                   [Link](https://arxiv.org/pdf/2402.13116)                    | [Link](https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs)  | 
| 2024  | **Navigate through Enigmatic Labyrinth A Survey of Chain of Thought Reasoning: Advances, Frontiers and Future**   |  ACL    |                   [Link](https://arxiv.org/abs/2309.15402)                    | -  | 
| 2024  | **Automatically Correcting Large Language Models: Surveying the Landscape of Diverse Automated Correction Strategies*   |  TACL    |                   [Link](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00660/120911)                    | -  | 

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

## Being a Thinking Model <span id="Thinking"></span>
### Supervised Fine-tuning (SFT)
#### Training Methods
| **Year**   | **Title**                                                                                     |  **Venue**    |                                       **Paper**                                            | **Code** |
| ---- |----------------------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------:|:----:|
| 2023  | **ALERT: Adapt Language Models to Reasoning Tasks**   |  ACL    |                   [Link](https://aclanthology.org/2023.acl-long.60.pdf)                    | -   |
| 2024  | **Abstraction-of-Thought Makes Language Models Better Reasoners**   |  EMNLP    |                   [Link](https://arxiv.org/pdf/2406.12442)                    | [Link](https://github.com/Raising-hrx/Abstraction-of-Thought)   |
| 2023  | **IMPLICIT CHAIN OF THOUGHT REASONING VIA KNOWLEDGE DISTILLATION**   |  -    |                   [Link](https://arxiv.org/pdf/2311.01460)                    | [Link](https://github.com/da03/implicit_chain_of_thought/)   |
| 2024  | **From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step**   |  -    |                   [Link](https://arxiv.org/pdf/2405.14838)                    | [Link](https://github.com/da03/Internalize_CoT_Step_by_Step)   |
| 2024  | **Training Large Language Models to Reason in a Continuous Latent Space**   |  -    |                   [Link](https://arxiv.org/pdf/2412.06769)                    | -  |
| 2024  | **Guiding Language Model Reasoning with Planning Tokens**   |  COLM    |                   [Link](https://arxiv.org/pdf/2310.05707)                    | [link](https://github.com/WANGXinyiLinda/planning_tokens)  |
#### Acquiring CoT 

### Reinforced Fine-tuning (RFT)
#### Training Methods
| **Year**   | **Title**                                                                                     |  **Venue**    |                                       **Paper**                                            | **Code** |
| ---- |----------------------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------:|:----:|
| 2022  | **Large Language Models Are Reasoning Teachers**   |  ACL    |                   [Link](https://arxiv.org/pdf/2212.10071)                    | [Link](https://github.com/itsnamgyu/reasoning-teacher)   |

#### Reward Modeling


## Being an Insight Model <span id="Insight-"></span>
#### Blue Hat (Planning)
| **Year**   | **Title**                                                                                     |  **Venue**    |                                       **Paper**                                            | **Code** |
| ---- |----------------------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------:|:----:|
| 2022  | **STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning**   |  NeurIPS    |                   [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/639a9a172c044fbb64175b5fad42e9a5-Paper-Conference.pdf)                    | [Link](https://github.com/ezelikman/STaR)  |

#### Green (Diverse thinking)

#### Red (Intuitive thinking)

#### Black (Reflection)


#### Yellow (Internal thinking)

#### White (Fact perception)

## Performance <span id="Performance"></span>
