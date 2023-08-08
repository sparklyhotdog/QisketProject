# QisketProject
### Project Description
Simulates a polarization entanglement distribution experiment factoring in practical issues.

This simulation models a polarization entanglement distribution experiment measuring pairs of entangled photons at different polarization basis states using two polarizers. It accounts for various real-world factors in a fiber-based quantum network, which include imperfect entanglement states from the photon source; optical loss, ambient light, and dispersion in fibers; as well as dark counts and timing jitter in single photon detectors. The simulation first utilizes the Qiskit library to calculate the expected photon pair rate at different bases and generates pairs of pseudo timestamp data sets accordingly. The modeling of effects such as optical loss and timing jitter is directly applied to these data sets by adding/dropping timestamps, or numerically changing their values. The simulation package then computes the cross-correlation between the timestamp data sets, which gives an estimation of the coincidence and accidental rate, as well as the coincidence to accidental ratio (CAR). From these results, we can evaluate the entanglement visibility of the distributed photons, which is an assessment of the quality of an entanglement state. Various plotting scripts are also provided in this simulation package which help to investigate how these factors affect the entanglement quality over different fiber lengths. The parameters are stored in a configuration file. The plots are saved in `simulations_with_practical_issues/plots`. For the most part, the parameters are saved in the title. I used some special characters (like $\lambda$ and $\tau$) to try to shorten the file names, which can unfortunately make copying and pasting the images more annoying. Some plots are from an earlier version of the plotting scripts, which had smaller axis labels.

There are also some ideal simulations coded in Jupyter Notebooks that use the Qiskit Library more extensively.

See the GitHub Wikipages for more information about the documentation. The section about the plotting scripts is still being worked on.

### Prerequisites
* numpy
* math
* matplotlib
* yaml
* [qiskit](https://qiskit.org/documentation/getting_started.html#installation)
* [alive_progress](https://pypi.org/project/alive-progress/)


