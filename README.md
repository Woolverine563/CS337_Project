
# TAMPo - CS337 Project

#### Authors: Ameya, Mridul, Param, Tanay

#### TF-C Paper: [Preprint](https://arxiv.org/abs/2206.08496)

Kindly run the following shell script that will download the datasets and place it in the appropriate directories:

1. `chmod 777 download_datasets.sh`
2. `./download_datasets.sh`

## Overview 

This repository now contains three processed datasets and the codes of a developed baseline model (TS-TCC), <strong><u>reproduced</u></strong> TF-C model, and our modifications made over it. TF-C is a novel pre-training approach for learning generalizable features that can be transferred across different time-series datasets. We evaluate TF-C (and our modifications) on two time series datasets with different sensor measurements and semantic meanings.

**TF-C approach -** our model has four components: a time encoder, a frequency encoder, and two cross-space projectors that map time-based and frequency-based representations, respectively, to the same time-frequency space. Together, the four components provide a way to embed the input time series to the latent time-frequency space such that time-based embedding and frequency-based embedding are close together. 
The TF-C property is realized by promoting the alignment of time- and frequency-based representations in the latent time-frequency space, providing a vehicle for transferring the well-trained model to a target dataset not seen before. During fine-tuning, we add a Cross-Entropy prediction loss to train the classifier along with fine-tuning the model.

**Modifications made -** instead of using a self-supervised contrastive loss even for fine-tuning as done in TF-C, we use a supervised contrastive loss; since during this phase, no generalisability is seeked for. These changes can be viewed in `trainer_plus.py`.

## Datasets

The scenarios contain electrodiagnostic testing and human daily activity recognition.

### Raw data

(1). **SleepEEG** contains 153 whole-night sleeping Electroencephalography (EEG) recordings that are monitored by sleep cassette. The data is collected from 82 healthy subjects.

(2). **Epilepsy** contains single-channel EEG measurements from 500 subjects. For each subject, the brain activity was recorded for 23.6 seconds.

(3). **Gesture** contains accelerometer measurements of eight simple gestures that differed based on the paths of hand movement.

The following table summarizes the statistics of all these eight datasets:

| Scenario # |              | Dataset      | # Samples    | # Channels | # Classes | Length | Freq (Hz) |
| ---------- | ------------ | ------------ | ------------ | ---------- | --------- | ------ | --------- |
| 1          | Pre-training | **SleepEEG** | 371,055      | 1          | 5         | 200    | 100       |
|            | Fine-tuning  | **Epilepsy** | 60/20/11,420 | 1          | 2         | 178    | 174       |
| 2          | Pre-training | **SleepEEG** | 371,055      | 1          | 5         | 200    | 100       |
|            | Fine-tuning  | **Gesture**  | 320/120/120  | 3          | 8         | 315    | 100       |

## Running the code

### Reproduce TS-TCC
You are advised to run the model from the corresponding folder under `TS-TCC` using the command-line patterns described by the original authors' `README.md` files whenever possible.

### Reproduce TF-C
There are three key arguments:
1. `mode` has two options *pretrain* or *fine_tune*.
2. `dataset` has three options *SleepEEG*, *Epilepsy* or *Gesture*. <strong>Kindly pretrain on *SleepEEG* and fine_tune on either *Epilepsy* or *Gesture*.</strong>

Now, we are ready to run the following commands to produce results:

```
python3 main.py --dataset SleepEEG --mode pretrain
```

```
python3 main.py --dataset Epilepsy --mode fine_tune
```

### Reproduce TAMPo
Essentially, the pretraining phase is the same as above. While fine-tuning, just add another argument `model` that takes either *TFC* (default) or *TAMPo*. Hence, run:

```
python3 main.py --dataset Epilepsy --mode fine_tune --model TAMPo
```
