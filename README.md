DAVE2Net – End-to-End Imitation Learning in CARLA



This repository implements an end-to-end autonomous driving pipeline using a DAVE-2–style convolutional neural network trained via imitation learning in the CARLA simulator.



The project focuses on behavioral cloning (BC) as a baseline and extends to DAgger (Dataset Aggregation) to study failure modes and generalization limits of end-to-end policies.



Overview



The goal of this project is to study how far a pure perception-to-control policy



𝑎 = 𝜋(𝑠)



can drive a vehicle in realistic simulated environments, and where it fails.



Key questions explored:

* How does dataset composition affect driving performance?
* What failure modes emerge in unseen states?
* How does DAgger mitigate covariate shift compared to vanilla BC?



Features

* CARLA-based expert data collection

&nbsp;     -  Traffic Manager autopilot

&nbsp;     - Optional PID expert

* Behavioral Cloning (BC) baseline

&nbsp;     - DAVE-2–style CNN

&nbsp;     - Supervised regression on steering / throttle / brake

* DAgger training

&nbsp;     - On-policy data aggregation

&nbsp;     - Iterative dataset expansion

* Evaluation \& analysis

&nbsp;     - Fixed-spawn evaluation protocol

&nbsp;     - Collision, stuck, and distance metrics

&nbsp;     - Post-hoc analysis scripts



Repository Structure

.

├── autopilot\_collect\_dave2\_min.py   # Expert data collection (CARLA autopilot)

├── dagger\_collect\_dave2.py           # DAgger data aggregation loop

├── expert\_pid.py                     # PID expert controller

│

├── Dataset\_Baseline.py               # BC dataset loader

├── Dataset\_DAgger.py                 # DAgger dataset loader

│

├── Network\_Baseline.py               # DAVE-2 style CNN

│

├── Training\_Baseline\_DAVE2.py         # Behavioral Cloning training

├── Training\_dagger\_DAVE2.py           # DAgger training

│

├── Drive\_dave2.py                    # Policy inference / driving

├── eval\_dave2.py                     # Evaluation script

├── analyze\_eval\_dave2.py             # Evaluation analysis

├── analyzing\_plot.py                 # Plotting utilities

│

├── .gitignore

├── README.md

└── LICENSE



DAVE-2 Network

The policy network follows the NVIDIA DAVE-2 architecture, consisting of:

* Convolutional layers for visual feature extraction
* Fully connected layers for control regression
* Direct mapping from RGB image → control commands



Output:

\[steering, throttle, brake]

Loss:

* Smooth L1 (Huber) regression



Training Pipeline

Behavioral Cloning (BC)

1. Collect expert demonstrations in CARLA

2\. Train DAVE-2 using supervised learning

3\. Evaluate on fixed spawn points



DAgger

1. Initialize policy from BC

2\. Roll out the learner policy

3\. Query expert for corrective actions

4\. Aggregate new data and retrain



Evaluation

Evaluation is performed using:

* Fixed spawn locations
* Multiple episodes per spawn
* Early termination on collision or stuck state



Metrics include:

* Total distance traveled
* Collision rate
* Stuck rate
* Success rate



Datasets \& Models

🚫 Datasets and trained model checkpoints are intentionally excluded from this repository.

Reason:

* Large size
* Reproducibility via scripts
* Clean research-oriented version control

All datasets can be regenerated using the provided data collection scripts.



Requirements

* Python 3.7+
* CARLA 0.9.x
* PyTorch
* NumPy, OpenCV, Matplotlib

(Exact environment setup depends on CARLA installation.)



Future Work

* Weather and traffic diversity
* Multi-sensor inputs (IMU, GNSS)
* Frame stacking \& data augmentation
* Offline RL baselines
* Comparison with classical control and RL agents



License



This project is licensed under the MIT License.



Acknowledgments

* NVIDIA DAVE-2
* CARLA Simulator
* Imitation Learning \& DAgger literature
