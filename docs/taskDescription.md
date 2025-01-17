# AI Engineer - time series

### Context

steepsoft is currently working on systems which perform sleep analysis in various ways. In order to generate the necessary data, we need a real-time sleep stage prediction system (REM, Deep sleep, light sleep, etc).

In order to generate this data using consumer grade electronics such as Apple Watches, until recently we have relied on [a paper that can be found here](https://academic.oup.com/sleep/article/42/12/zsz180/5549536?login=false).

[The paper’s codebase can be found here](https://github.com/ojwalch/sleep_classifiers), but it is unnecessarily complex. Therefore, **we are on a mission to simplify it**, make it more efficient and create an AI-powered mechanism to predict the sleep stage that a user is in, in real time.

<aside>
❗ The dataset you have to use for this problem can be found here: https://physionet.org/content/sleep-accel/1.0.0/ and it contains the following data points:
- heart rate → heart rate while sleeping
- motion (x, y, z) → motion of the wrist while sleeping
- steps → number of steps the person walked the day prior to their sleep session
- labels (sleep labels) → REM, non-rem, Deep Sleep, Light Sleep

You will find more information on this dataset in the paper.

</aside>

### The challenge

We need a simple Python-based system that is able to train a Neural Network architecture (mentioned below) and then once trained use it to predict results on a sample from the dataset to assess accuracy. Train the model on 90% of data, and then assess its accuracy on the rest of 10% of the data. In the end, we want to be able to pass it heart rate , motion , steps and get a label as a response. **There’s no expectation of accuracy, although common literature will position the potential accuracy at about ~70%.** Anything beyond is considered a bonus point.

Since this problem involves time series data ([read about time series here](https://en.wikipedia.org/wiki/Time_series)), we expect that **Transformer architectures**, **Recurrent Neural Networks** or **Long Short Term Memory Networks** will perform best, as they take into account also the states prior to the current t[i] point in time.

<aside>
💡 For example, a person is more likely to be in the deep  state rather than awake  state if their previous state was light . That’s why having prior states is so important.

</aside>

<aside>
❗ We do not expect you to write Neural Network implementations from scratch! Please use libraries and frameworks such as: sklearn , tensorflow, keras or pytorch .  Choose your weapon! 🔫
—

**Transformer architecture:**

https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html

https://arxiv.org/abs/2205.01138

https://www.nature.com/articles/s41598-024-76197-0

https://arxiv.org/pdf/2105.11043

https://www.medrxiv.org/content/10.1101/2022.11.21.22282544v1.full.pdf

**LSTM architecture:**
Example implementation of LSTM here: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

https://github.com/OxWearables/asleep/blob/main/src/asleep/sleepnet.py

Use it thoughtfully and adapt/improve it based on the challenge at hand.

[Here’s also some theoretical documentation](https://cnvrg.io/pytorch-lstm/) you can use to your advantage.

</aside>

<aside>
💡

If you’re confident in a different direction or idea and can prove better reliability and performance, we’re all in for it! Your opinion is valued and we will assess it accordingly.

</aside>

### Expected deliverables

We expect the following deliverables:

1. source code to the training & inference system;
2. documentation which will explain how to run the system, how to test the system, and what are the results; also the documentation of the thought process, assumptions and ideas to validate;

   <aside>
   💡 Keep in mind that we are trying to understand your approach to solving technical problems, and not just coding skills.

   </aside>

3. the ability to run the system on any other machine (replicability can be achieved either by using venv, conda, or docker-preferred)
4. if completed, submit accuracy results

### Bonus points

- Accuracy of the REM sleep stage over 70%;
- docker environment
- tests & data validation
- result replication and presentation across multiple train & validation runs on randomized train/validation sets
  - generate train/validation datasets and run the system
  - do it for 5 more times
  - present results of each run → results should be consistent with initial findings

### Extras

- You can use any resource available to you (Google, ChatGPT, documentation, etc);
- Even if not completed successfully, we would like to understand what the progress was and what the blockers were;

### 💡 Reminder

- We value code quality! Use best practices, show your coding skills, but also coding maturity with production-level code.
- Make sure you use libraries that make your life easier. We love seeing hackers show us their tool stack.
- It’s not about ticking every box, but rather about ticking the right ones at a high enough standard.
