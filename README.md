# TDTracker: Event-based Eye Tracking Challenge - CVPR 2025 - Third Place Solution
Exploring Temporal Dynamics in Event-based Eye Tracker

### Usage
1. Prepare the data:

        cd dataprocess
        python 3et_plus.py

2. Put the train_aug.h5 and test_aug.h5 to ./data/xxx/:

3. Modify data_path, log_name and others.

4. Run the train script:
    
        python train.py
5. Run the test script to generate initial CSV (MSE: 1.5532):

        python test.py

6. Run the test script to generate initial CSV (MSE: 1.4932):

        python post_process.py
