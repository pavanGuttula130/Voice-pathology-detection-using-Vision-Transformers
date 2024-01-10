# Voice-pathology-detection-using-Vision-Transformers
Voice pathology detection using VMD (Variational Mode Decomposition) and Vision Transformers and Machine Learning
This repository contains the code and resources for a project focused on voice pathology detection using two different approaches: Machine Learning and Deep Learning.

## Overview

The project aims to detect voice pathology by analyzing vocal recordings of the vowel /a/. Two distinct approaches were employed: Machine Learning and Deep Learning.

### Dataset

The vocal recordings used in this project were sourced from the SVD database. The dataset comprises voices of the vowel /a/ and is considered a benchmark in the field. Within the SVD database, three voice classes were selected:

1. **Healthy**: Voices belonging to the healthy class.
2. **Reinke Odem**: Voices belonging to the pathological class known as Reinke Odem.
3. **Laryngitis**: Voices belonging to the class associated with laryngitis.

## Machine Learning Approach

In the Machine Learning approach, the following steps were followed:

1. **Feature Extraction**: Fundamental features of the voices, such as fundamental frequency (f0), Jitter, Shimmer, and 39 MFCC (Mel Frequency Cepstral Coefficients) features, were extracted. These features were concatenated into an array.

2. **Classification**: The concatenated feature arrays were then used as input for classification models. The following models were employed:
   - LGBM (Light Gradient Boosting Machine)
   - XGBoost
   - Extra Tree Classifier

## Deep Learning Approach

In the Deep Learning approach, the following steps were undertaken:

1. **VMD Mode Extraction**: The first step involved extracting Voice Mode Decomposition (VMD) modes from the voices. In this experiment, three modes were chosen to eliminate noise in higher frequencies.

2. **Combination of Modes**: The three extracted modes were combined.

3. **Mel-Spectrogram Generation**: Mel-spectrograms were generated for each voice using the combined VMD modes. These mel-spectrograms serve as the input data for the Deep Learning model.

4. **Classification**: Vision Transformers were employed to classify the set of voices based on the generated mel-spectrograms.


## Usage

To use this codebase, follow these steps:

1. Prepare your voice dataset in the `data/` directory.
2. Seperate the train data and test data.
3. Prepare a csv file such that it should contains names of all audio files and the pathology associated with the voice
4. Use `VMD-Melspectrogram.py/` file to generate Mel-spectograms.
5. Change the paths of the folders and csv file in `Voice-pathology-detection.py/` file and run python file.


If you have any questions or suggestions, feel free to contact the project contributors:

- Onteddu Chaitanya Reddy - chaitanyareddy0702@gmail.com
- Illa Dinesh Kumar - idineshkumar12321@gmail.com
- Guttula Pavan - pavanguttula123@gmail.com
- Pillalamarri Akshaya - akshayapillalamarri213@gmail.com
-  Anirudh Edpugnati - aniedpuganti@gmail.com

---

