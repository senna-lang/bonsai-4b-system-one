# Public accuracy benchmark: protocol

This protocol fixes the datasets, prompts, option order, and metrics; they were not changed after seeing results.

## System

Frozen Ternary-Bonsai-4B, no adapter, with the prompt in [METHOD.md](METHOD.md), one canonical option order (the order listed below), and answer-letter readout.

## Tasks

500 rows per task, sampled with `dataset.shuffle(seed=0).select(range(500))` from the split below.

| Task | Hugging Face dataset | Split | Kind | State | Question | Options (in order) |
| --- | --- | --- | --- | --- | --- | --- |
| AG News | `fancyzhx/ag_news` | test | choice | article text | What is the topic of this news article? | World, Sports, Business, Sci/Tech |
| MASSIVE scenario | `mteb/amazon_massive_scenario` (`en`) | test | choice | utterance | Which scenario does this request to a voice assistant belong to? | the 18 scenario labels, alphabetical |
| MNLI | `nyu-mll/glue` (`mnli`) | validation_matched | choice | `Premise: …` / `Hypothesis: …` | What is the relationship between the premise and the hypothesis? | entailment, neutral, contradiction |
| BoolQ | `google/boolq` | validation | noul | passage | the dataset question, capitalized, with `?` | No, Yes |
| SST-5 | `SetFit/sst5` | test | score | sentence | How positive is the sentiment of this text? | very negative, negative, neutral, positive, very positive |

Option labels are the datasets' own label names.

## Metrics

- Accuracy of the highest-probability option, with a 95% bootstrap interval (2,000 resamples, seed 0).
- SST-5 additionally reports mean absolute error in levels.

## Data provenance

Nothing in this repository is trained. The prompt format and readout were designed on other data (AMI/ICSI via QMSum, GitHub issues, NASA ASRS, NHTSA, public regulations) before these benchmarks were run.

