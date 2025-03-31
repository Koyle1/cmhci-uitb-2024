This repository contains the research conducted by Felix Coy form October 2024 to March 2025 during the course Computational Modeling for Human-Computer-Interaction.
The research builds on the user-in-the-box framework, adding research specific code.
To run the code, please follow the instructions containied in the original user-in-the-box README (here UITB_README.md).

This repository adds the following repositories/files:
(1) /uitb/bm_models/mobl_arms_index_EMG_Variance: biomechanical model which applies noise based on EMG signal Variance to the motor activation patterns
(2) /uitb/bm_models/mobl_arms_index_fuzzy: biomechanical model which applies noise based on gaussian distribution to the motor activation patterns
(3) /uitb/perception/vision/fixed_eye_blurry: perception model that applies gaussian blur before passing the image to the rl model
(4) /uitb/tasks/pointing: Contains the task used for evaluating the different models

(5) /simulators/mobl_arms_index_pointing: trained model without noise
(6) /simulators/mobl_arms_index_pointing_with_blurry_vision: trained model with (3)
(7) /simulators/mobl_arms_index_pointing_with_emg_variance: traind model with (1)
(8) /simulators/mobl_arms_index_pointing_with_emg_variance_blurry_vision: trained model with (1) and (3)
(9) /simulators/mobl_arms_index_pointing_with_motor_noise: trained model with (2)
(10) /simulators/mobl_arms_index_pointing_with_motor_noise_blurry_vision: trained model with (2) and (3)


