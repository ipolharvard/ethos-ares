# ETHOS - EHR foundation model

This repository implements Adaptive Risk Estimation System (ARES) for Hospital Mortality, ICU
Admission, Prolonged Length of Stay, and Composite (HM+IU+PLoS). In addition, it contains all the
experiments conducted in our paper ([preprint](https://arxiv.org/abs/2502.06124)). It builds on our
previous work on EHR foundation models by completely
reimplementing [ETHOS](https://www.nature.com/articles/s41746-024-01235-0) (formerly
available [here](https://github.com/ipolharvard/ethos-paper))
to achieve higher performance, improved usability, robustness, and further expand our work.

## Features

- In this release, we demonstrate that ETHOS outperforms the baselines presented in our paper and
  introduce ARES, a method for explaining patient health trajectories. We test ETHOS on the MIMIC-IV
  with the MIMIC-IV-ED extension dataset on all most common tasks in the domain.
- A key feature of ETHOS is zero-shot prediction, where known patient health trajectories are
  extrapolated to generate future patient health trajectories (fPHT). This process is repeated to
  derive probability estimates.
- The current main model architecture that we use is GPT2 (no bias). Feel free to experiment
  with architectures specifically suited for EMR.
- ETHOS is not a natural language model. It uses a specialized language designed to succinctly
  describe patient health events, which we believe helps the model learn better representations and
  token relationships.
- This implementation uses [MEDS](https://github.com/Medical-Event-Data-Standard/meds) as an
  intermediate data representation.
- We provide a full pipeline that includes tokenization, training, and inference.
- We invite everyone to a discussion in the Issues section of this repository.

<p align="center">
<a href="https://arxiv.org/abs/2502.06124">
  <img src="./figures/ethos_ares_workflow.png" width="70%" alt="ETHOS-ARES workflow">
</a>
</p>

## Paper reproducibility

We provide the complete code necessary to reproduce all experiments presented in the paper.

Additionally, all precomputed inference results of our experiments are available in
`results.tar.gz` [[Google Drive (1.8GB)]](https://drive.google.com/file/d/1P2y70iZO3ZbwkROVCzJa7FCubQvk1qsE/view?usp=sharing).
Once unpacked in the project's root directory, this file will allow generating
all the figures in the paper using the notebooks: `notebooks/figures.ipynb` and
`notebooks/trajectories.ipynb`. The test set on which these results were inferred can be easily
recreated by running the MEDS extraction and tokenization pipelines (see below). MEDS guarantees the
same data split if run on the same data with the same configuration (see `scripts/meds`).

We do not publish the tokenized dataset or the pretrained model due to restrictions on MIMIC
derivatives, and we conclude that these components are not suitable for publication on PhysioNet.

## Workflow

Package entry points:

1. `ethos_tokenize` - Example in `scripts/run_tokenization.sh`.
2. `ethos_train` - Example in `scripts/run_training.sh`.
3. `ethos_infer` - Example in `scripts/run_inference.sh`.

## Installation

[Optional] We strongly encourage the use of a virtual environment, for example, Conda:
To create a new conda env:

```bash
conda create --name ethos python=3.12
conda activate ethos
```

Fetch the project and set it up in the development mode (`-e`) and install all necessary
dependencies for running notebooks and scripts by executing:

```bash
git clone https://github.com/ipolharvard/ethos-ares
cd ethos-ares
pip install -e .[jupyter]
```

## ETHOS tokenization guide

ETHOS tokenization uses an intermediate
format [MEDS](https://github.com/Medical-Event-Data-Standard/meds), extracted via
the [MEDS_transforms](https://github.com/mmcdermott/MEDS_transforms) pipeline. Scripts for running
this pipeline are located in `scripts/meds`.

Below is an example command to run the extraction (where `$suffix` should be "ed" or empty). In case
of `ed`, it requires the MIMIC-IV-ED extension to be present in the input directory:

```bash
export N_WORKERS=7

# Please define output_dir, strogly suggested is <PROJECT_ROOT>/data
suffix="ed" # or ""

bash run_mimic.sh \
    "$MIMIC_IV_DIR" \
    "$output_dir/mimic-2.2-premeds$suffix" \
    "$output_dir/mimic-2.2-meds$suffix" \
    "$suffix"
```

Note, that using 7 workers for the tokenization requires around 250GB of RAM peak usage. You can
reduce the memory requirement by reducing the number of workers.

In the [paper](https://arxiv.org/abs/2502.06124), data is split into 90% training and 10% testing.
It can be adjusted in the `scripts/meds/mimic/configs/extract_MIMIC.yaml` file. Note, that keys:
`train`, `tuning` and `held_out` have to be always present in the config file, but can be set to
null.

```yaml
split_and_shard_subjects:
    ...
    split_fracs:
        train: 0.9
        test: 0.1
        tuning: null
        held_out: null
```

Once data extraction is complete, you can tokenize using the `ethos_tokenize` command, demonstrated
in `scripts/run_tokenization.sh`. Ensure the file hierarchy matches what the script expects, or
modify the script accordingly before running.

## Cite us

If you use ETHOS or ETHOS-ARES in your research, please cite our work:

[[1]](https://arxiv.org/abs/2502.06124)
Renc, P., Grzeszczyk, M. K., Oufattole, N., Goode, D., Jia, Y., Bieganski, S., ... & Sitek, A. (
2025).
Foundation Model of Electronic Medical Records for Adaptive Risk Estimation. arXiv preprint arXiv:
2502.06124.

```
@misc{renc2025ehrfoundationmodel,
      title={Foundation Model of Electronic Medical Records for Adaptive Risk Estimation},
      author={Pawel Renc and Michal K. Grzeszczyk and Nassim Oufattole and Deirdre Goode and Yugang Jia and
      Szymon Bieganski and Matthew B. A. McDermott and Jaroslaw Was and Anthony E. Samir and Jonathan W.
      Cunningham and David W. Bates and Arkadiusz Sitek},
      year={2025},
      eprint={2502.06124},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.06124},
}
```

[[2]](https://www.nature.com/articles/s41746-024-01235-0)
Pawel Renc, Yugang Jia, Anthony E Samir, Jaroslaw Was, Quanzheng Li, David W Bates, Arkadiusz Sitek,
"Zero shot health trajectory prediction using transformer" npj Digital Medicine, 19 Sep 2024

```
@article{renc_zero_2024,
	title = {Zero shot health trajectory prediction using transformer},
	volume = {7},
	copyright = {2024 The Author(s)},
	issn = {2398-6352},
	url = {https://www.nature.com/articles/s41746-024-01235-0},
	doi = {10.1038/s41746-024-01235-0},
	abstract = {Integrating modern machine learning and clinical decision-making has great promise for mitigating healthcare’s increasing cost and complexity. We introduce the Enhanced Transformer for Health Outcome Simulation (ETHOS), a novel application of the transformer deep-learning architecture for analyzing high-dimensional, heterogeneous, and episodic health data. ETHOS is trained using Patient Health Timelines (PHTs)—detailed, tokenized records of health events—to predict future health trajectories, leveraging a zero-shot learning approach. ETHOS represents a significant advancement in foundation model development for healthcare analytics, eliminating the need for labeled data and model fine-tuning. Its ability to simulate various treatment pathways and consider patient-specific factors positions ETHOS as a tool for care optimization and addressing biases in healthcare delivery. Future developments will expand ETHOS’ capabilities to incorporate a wider range of data types and data sources. Our work demonstrates a pathway toward accelerated AI development and deployment in healthcare.},
	language = {en},
	number = {1},
	urldate = {2024-09-24},
	journal = {npj Digital Medicine},
	author = {Renc, Pawel and Jia, Yugang and Samir, Anthony E. and Was, Jaroslaw and Li, Quanzheng and Bates, David W. and Sitek, Arkadiusz},
	month = sep,
	year = {2024},
	pages = {1--10},
}
```
