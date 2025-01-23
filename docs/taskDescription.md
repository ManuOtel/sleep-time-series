# Sleep Stage Classification System

### Context

This project focuses on developing a real-time sleep stage prediction system (REM, Deep sleep, light sleep, etc) using consumer-grade wearable devices like Apple Watches. Previously, the approach was based on [this research paper](https://academic.oup.com/sleep/article/42/12/zsz180/5549536?login=false).

[The original codebase](https://github.com/ojwalch/sleep_classifiers) is unnecessarily complex. The goal is to simplify it, make it more efficient and create an AI-powered mechanism to predict sleep stages in real-time.

<aside>
❗ The dataset for this project is available at: https://physionet.org/content/sleep-accel/1.0.0/ and contains:
- heart rate → heart rate while sleeping
- motion (x, y, z) → motion of the wrist while sleeping  
- steps → number of steps walked the day prior to sleep
- labels (sleep labels) → REM, non-rem, Deep Sleep, Light Sleep

More information on this dataset can be found in the paper.

</aside>

### Requirements

The system should be Python-based and able to train a Neural Network architecture to predict sleep stages. The model should be trained on 90% of the data and evaluated on the remaining 10%. The system should accept heart rate, motion, and steps data as input and output predicted sleep stage labels. While common literature suggests accuracy around ~70%, any improvements beyond this are welcome.

Since this involves time series data ([read about time series here](https://en.wikipedia.org/wiki/Time_series)), **Transformer architectures**, **Recurrent Neural Networks** or **Long Short Term Memory Networks** are recommended as they consider states prior to the current t[i] point in time.

<aside>
💡 For example, a person is more likely to be in the deep state rather than awake state if their previous state was light. That's why having prior states is so important.

</aside>

<aside>
❗ The implementation should leverage existing libraries and frameworks like sklearn, tensorflow, keras or pytorch.

**Transformer architecture resources:**

https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html

https://arxiv.org/abs/2205.01138

https://www.nature.com/articles/s41598-024-76197-0

https://arxiv.org/pdf/2105.11043

https://www.medrxiv.org/content/10.1101/2022.11.21.22282544v1.full.pdf

**LSTM architecture resources:**
Example implementation: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

https://github.com/OxWearables/asleep/blob/main/src/asleep/sleepnet.py

[Additional theoretical documentation](https://cnvrg.io/pytorch-lstm/)

</aside>

<aside>
💡 Alternative approaches that can demonstrate better reliability and performance are welcome.

</aside>

### Deliverables

1. Source code for the training & inference system
2. Documentation covering:
   - System setup and execution instructions
   - Testing procedures
   - Results analysis
   - Documentation of methodology, assumptions and validation approaches

3. System replicability through venv, conda, or docker (preferred)
4. Accuracy results and analysis

### Additional Features

- REM sleep stage accuracy exceeding 70%
- Docker environment
- Tests & data validation
- Result replication across multiple training runs:
  - Generate multiple train/validation splits
  - Execute 5+ complete training runs
  - Present consistent results across runs