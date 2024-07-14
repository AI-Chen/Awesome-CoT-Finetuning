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
| 2024  | **Turning Dust into Gold: Distilling Complex Reasoning Capabilities from LLMs by Leveraging Negative Data**   |  AAAI    |                   [Link](https://ojs.aaai.org/index.php/AAAI/article/view/29821)                    | [Link](https://github.com/Yiwei98/TDG)   |

## Self-Enhancement <span id="Self-Enhancement-"></span>
| **Year**   | **Title**                                                                                     |  **Venue**    |                                       **Paper**                                            | **Code** |
| ---- |----------------------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------:|:----:|
| 2022  | **Large Language Models Can Self-Improve**   |  ACL    |                   [Link](https://aclanthology.org/2023.emnlp-main.67/)                    | -  |

## Application <span id="Applicationt-"></span>
| **Year**   | **Title**                                                                                     |  **Venue**    |                                       **Paper**                                            | **Code** |
| ---- |----------------------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------:|:----:|
| 2024  | **Effective Distillation of Table-based Reasoning Ability from LLMs**   |  ACL    |                   [Link](https://aclanthology.org/2024.lrec-main.492/)                    | [Link]([https://github.com/Yiwei98/TDG](https://github.com/Bernard-Yang/DistillTableCoT))  |
