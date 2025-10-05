# Awesome-CoT-Finetuning

A collection of papers and datasets for the chain of thought finetuning (CoT-Finetuning) survey. We are looking forward to other participants to share their papers and codes. If interested, please contact chenxs@nudt.edu.cn or xschenranker@gmail.com. :fire: :fire: :fire: 

- Comprehensive Review: We comprehensively investigate typical CoT fine-tuning methods based on a bi-level taxonomy, i.e., top-level (Six Thinking Hats), and base-level (techniques), which offers a novel perspective on CoT fine-tuning and facilitates the understanding of the developmental trajectories among different CoT fine-tuning methods.
- Insightful Analysis: Based on the Six Thinking framework, we analyze the strengths and limitations of existing CoT fine-tuning methods in enabling LLMs to develop corresponding reasoning abilities, which provides valuable guidance for researchers in selecting an appropriate baseline for their research.
- Potential Opportunity: Building upon the Six Thinking Hats framework, we identify and summarize the key challenges currently faced by CoT fine-tuning and point out some potential opportunities that will inspire future studies.
- Open-source Resource: We will keep this GitHub repository continuously updated for researchers to track the latest developments.

:bell: :bell: :bell: Update at Oct 2025

![Example](./evolution.png)

# Bookmarks
- [Survey Papers](#survey-papers-)
- [Datasets](#datasets-)
- [Being a Thinking Model](#Thinking-)
- [Being an Insight Model](#Insight-)

## Survey Papers <span id="survey-papers-"></span>
| **Year** | **Title** | **Venue** |
|------|-------|-------|
| 2023 | [Towards Reasoning in Large Language Models: A Survey](https://arxiv.org/abs/2212.10403 ) | ACL | 
| 2025 | [Beyond chain-of-thought: A survey of chain-of-X paradigms for LLMs](https://aclanthology.org/2025.coling-main.719/) | ICCL | 
| 2024 | [Navigate through enigmatic labyrinth a survey of chain of thought reasoning: Advances, frontiers and future](https://aclanthology.org/2024.acl-long.65/) | ACL | 
| 2025 |  [Reasoning Language Models: A Blueprint](https://arxiv.org/abs/2501.11223 ) | Arxiv |
| 2025 | [Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models](https://arxiv.org/abs/2501.09686 )  | Arxiv | 
| 2025 | [Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models](https://arxiv.org/abs/2503.09567 ) | Arxiv |  
| 2025 |  [From System 1 to System 2: A Survey of Reasoning Large Language Models](https://arxiv.org/abs/2502.17419) | Arxiv | 
| 2025 | [LLM Post-Training: A Deep Dive into Reasoning Large Language Models](https://arxiv.org/abs/2502.21321 ) | Arxiv | 
| 2025 |  [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models (arxiv.org)](https://arxiv.org/abs/2503.16419) | Arxiv |             

## Datasets <span id="datasets-"></span>
| **Category**   | **Dataset**   | **Train** | **Valid** | **Test** | **Task Description**                                | **Rationale (ie. CoT)** | **Dataset Download** | **Paper** |
|----------------|---------------|-----------|-----------|----------|-----------------------------------------------------|-------------------------|----------------------|-----------|
| General Task   | BBH           | 0         | 0         | 6,511    | 23 common reasoning tasks                           | No                      | [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard) | [BBH](https://aclanthology.org/2023.findings-acl.824/) |
|                | SuperGPQA     | 0         | 0         | 26,529   | Graduate-level Q&A across 285 disciplines          | No                      | [SuperGPQA](https://supergpqa.github.io/) | [SuperGPQA](https://arxiv.org/abs/2502.14739) |
|                | MMLU          | 0         | 1,540     | 14,368   | Multiple-choice Q&A across 57 tasks                | No                      | [MMLU](https://github.com/hendrycks/test) | [MMLU](https://arxiv.org/pdf/2009.03300) |
|                | MMLUPro       | 0         | 70        | 12,032   | Q&A tasks across 285 disciplines                    | Partial                 | [MMLUPro](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro) | [MMLUPro](https://arxiv.org/pdf/2406.01574) |
| Mathematics    | GSM8K         | 7,500     | 0         | 1,000    | Grade school math                                    | Yes                     | [GSM8K](https://github.com/openai/grade-school-math) | [GSM8K](https://arxiv.org/abs/2110.14168) |
|                | MGSM          | 88        | 0         | 2,750    | Multilingual version of GSM8K                        | Yes                     | [MGSM](https://github.com/google-research/url-nlp) | [MGSM](https://openreview.net/forum?id=fR3wGCk-IXp) |
|                | AQuA          | 100,949   | 250       | 250      | Algebraic word problems                              | Yes                     | [AQuA](https://github.com/deepmind/AQuA) | [AQuA](https://aclanthology.org/P17-1015/) |
|                | MATH          | 7,500     | 0         | 5,000    | Competitive math                                     | Yes                     | [MATH](https://github.com/hendrycks/apps) | [MATH](https://arxiv.org/pdf/2103.03874) |
|                | AIME          | 0         | 0         | Updated annually | American invitational mathematics examination | Yes                     | [AIME](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions) | [AIME](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions) |
|                | Geometry3K    | 2,101     | 300       | 601      | Geometry Problem with symbolic reasoning             | No                      | [Geometry3K](https://lupantech.github.io/inter-gps) | [Geometry3K](https://aclanthology.org/2021.acl-long.528/) |
| Coding         | CodeContests  | 13,328    | 117       | 165      | Competitive programming                              | No                      | [CodeContests](https://www.science.org/doi/10.1126/science.abq1158) | [CodeContests](https://www.science.org/doi/abs/10.1126/science.abq1158) |
|                | LiveCodeBench | 0         | 0         | Continuous updates | Periodically updated programming             | No                      | [LiveCodeBench](https://livecodebench.github.io/) | [LiveCodeBench](https://openreview.net/forum?id=chfJJYC3iL) |
|                | MHPP          | 0         | 0         | 210      | Manually created Python programming                  | No                      | [MHPP](https://github.com/SparksofAGI/MHPP) | [MHPP](https://openreview.net/forum?id=TVFVx8TUbN) |
|                | EquiBench     | 0         | 0         | 2,400    | Equivalence checking for two programs                | No                      | [EquiBench](https://github.com/Anjiang-Wei/equibench) | [EquiBench](https://doi.org/10.48550/arXiv.2502.12466) |
|                | MBPP Pro      | 0         | 0         | 378      | Self-invoking code generation                        | No                      | [MBPP Pro](https://github.com/CodeEval-Pro/CodeEval-Pro) | [MBPP Pro](https://aclanthology.org/2025.findings-acl.686/) |
|                | SWEbench      | 19,000    | 0         | 2,294    | Solving real-world GitHub issues                     | No                      | [SWEbench](https://www.swebench.com/) | [SWEbench](https://openreview.net/forum?id=VTF8yNQM66) |
|                | BFCL v3       | 0         | 0         | 2,000    | Function-calling tasks                               | No                      | [BFCL v3](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#benchmarking) | [BFCL v3](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#benchmarking) |
| Commonsense    | HotpotQA      | 90,564    | 7,405     | 14,810   | Reading comprehension                                | No                      | [HotpotQA](https://HotpotQA.github.io) | [HotpotQA](https://aclanthology.org/D18-1259/) |
|                | CommonsenseQA | 9,797     | 1,225     | 1,225    | Multiple-choice Q&A                                  | No                      | [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) | [CommonsenseQA](https://aclanthology.org/N19-1421/) |
|                | StrategyQA    | 2,290     | 0         | 490      | True or false Q&A                                    | Yes                     | [StrategyQA](https://allenai.org/data/strategyqa) | [StrategyQA](https://aclanthology.org/2021.tacl-1.21/) |
|                | OpenBookQA    | 13,328    | 117       | 165      | Multiple-choice Q&A                                  | No                      | [OpenBookQA](http://data.allenai.org/OpenBookQA) | [OpenBookQA](https://aclanthology.org/D18-1260/) |
| Domain Knowledge| MedQA         | 48,876    | 6,109     | 6,112    | Multilingual multiple-choice medical Q&A             | No                      | [MedQA](https://github.com/jind11/MedQA) | [MedQA](https://arxiv.org/pdf/2009.13081) |
|                | JAMA          | 0         | 0         | 1,524    | Multiple-choice clinical Q&A                         | Yes                     | [JAMA](https://github.com/HanjieChen/ChallengeClinicalQA) | [JAMA](https://aclanthology.org/2025.naacl-long.182/) |
|                | MedXpertQA    | 0         | 0         | 4,460    | Multimodal multiple-choice medical Q&A               | Yes                     | [MedXpertQA](https://github.com/TsinghuaC3I/MedXpertQA) | [MedXpertQA](https://openreview.net/forum?id=IyVcxU0RKI) |
|                | GPQA          | 0         | 0         | 448      | Biology, physics, and chemistry multiple-choice Q&A  | Yes                     | [GPQA](https://github.com/idavidrein/gpqa/) | [GPQA](https://openreview.net/forum?id=Ti67584b98) |
|                | Zebralogic    | 0         | 0         | 1,000    | Logic grid puzzles                                   | No                      | [ZebraLogic](https://hf.co/spaces/allenai/ZebraLogic) | [ZebraLogic](https://openreview.net/forum?id=sTAJ9QyA6l) |
| Others         | ToolBench     | 0         | 0         | 1,524    | General tool-use tasks                               | Yes                     | [ToolBench](https://github.com/OpenBMB/ToolBench) | [ToolBench](https://arxiv.org/pdf/2307.16789) |
|                | ALFWorld      | 3,553     | 0         | 274      | 6 types of decision making tasks                     | No                      | [ALFWorld](https://alfworld.github.io/) | [ALFWorld](https://openreview.net/forum?id=0IOX0YcCdTn) |
|                | ChartQA-H     | 7,398     | 960       | 1,250    | Charts with visual and logical reasoning             | No                      | [ChartQA-H](https://github.com/vis-nlp/ChartQA) | [ChartQA-H](https://aclanthology.org/2022.findings-acl.177/) |
|                | ChartQA-M     | 20,901    | 960       | 1,250    | Charts with visual and logical reasoning             | No                      | [ChartQA-M](https://github.com/vis-nlp/ChartQA) | [ChartQA-M](https://aclanthology.org/2022.findings-acl.177/) |

## Being a Thinking Model <span id="Thinking-"></span>
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

![Example](./performance.png)
