# Activity-Classification-Challenge
#### The project is to build your own complete system for activity classification from raw sensor data sampled at the rate of 50 Hz. You have freedom in designing your own feature extraction method, choosing an appropriate type of classifier, and exploiting the given data set for system development.

## Main description

#### 1. Scenario
The accelerometer and gyroscope in a smartphone can be used to monitor the user’s activity such as walking,
running, etc. There are numerous applications based on activity monitoring. For instance, a fitness application
can report the daily amount of physical exercise by measuring how long the user walked or ran during the day.
In this project, the following six activities are considered:

    • Class 1: Walking
    • Class 2: Walking upstairs
    • Class 3: Walking downstairs
    • Class 4: Sitting
    • Class 5: Standing
    • Class 6: Laying

#### 2. Task and evaluation
This project is a competition, meaning that the accuracy of your system on a test data set will be considered in
the final mark.

Your final system is composed of two modules, one for training and another for test. The training module
should be named as activity_train.py (or some other proper extension), and will read the training data,
perform training of a model, and store the trained model. The test module should be named as
activity_test.py (or some other proper extension), and will read the test data, load the trained model, perform
classification, store the classification result.

Do not mix the training and test modules. In particular, the test module should only perform testing using the
trained model, and should not perform model training. In addition, the test module should perform
classification of each data file separately without using any information of other test data files.
While you will come up with one final system, you will propose two (or more) different systems (using
different models), and choose one of them as your final system.

Do not exploit external data other than the provided data. Pre-trained models are not allowed, either.

#### 3. Training data set
The data set for system development is composed of several text files. Each file contains a 128x6 matrix,
corresponding to data from the six sensors measured within a time window having a length of 128 samples
(i.e., 2.56 sec). The six columns represent accelerations (in standard gravity unit g=9.8m/s^2) in X, Y, and Z
directions, and angular velocities (in rad/sec) in X, Y, and Z directions, respectively. The file name of a file is
2
“aaaa_bb_c.txt,” where aaaa is the file ID, bb is the user ID, and c is the ground truth activity class (from 1 to
6).

#### 4. Test data set
The test data set will be released later. Only 6-dimensional input values will be given without any information
about the user ID and ground truth activity type. The file name will be thus “aaaa.txt,” where aaaa is the file
ID. Note that the users do not overlap between the training and test data.

#### 5. Classification result
You will submit one text file named “result_honggildong.txt” (i.e., your name in English), containing the
classification result for each file (from 1 to 6) in each row. It will be compared to the (undisclosed) ground
truth activity class and the accuracy will be measured.

## Requirements

- Ubuntu 16.04
- CUDA 10.1
- cuDNN 7.5
- Python 3.6
- sklearn 0.15.0.
- numpy 1.15.4
- matplotlib 2.1.0

## Training

```bash
python3 activity_train.py
```

## Testing

```bash
python3 activity_test.py
```