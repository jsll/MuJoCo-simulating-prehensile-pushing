# Simulating prehensile pushing

This repository contains the MuJoCo simulation environment used for doing the prehensile pushing experiment in the paper [Pushing Everything Everywhere All At Once: Probabilistic Prehensile Pushing
](https://arxiv.org/pdf/2503.14268). 


https://github.com/user-attachments/assets/adcbe9ad-5d92-4fc7-a5f4-5bf083ac45b2


## Installation

```bash
conda create -f environment.yml
conda activate simulating_prehensile_pushing
```

## Usage

Run

```bash
python simulate.py
```

If you want to generate new trajectories then please a look at the [following code](https://github.com/PatrizioPerugini/Probabilistic_prehensile_pushing).

To see optional arguments run
```bash
python simulate.py -h
```

```shell
options:
  -h, --help                        Show this help message and exit
  --filename FILENAME               Path to the experiment file.
  --max_velocity MAX_VELOCITY       Maximum allowed joint velocity in radians/second.
  --no_control                      Run the simulation without any control.
  --damping DAMPING                 Damping term for the pseudoinverse. This is used to prevent joint velocities from becoming too large when the Jacobian is close to singular.
  --integration_dt INTEGRATION_DT   Integration time-step in seconds. This corresponds to the amount of time the joint velocities will be integrated for to obtain the desired joint positions.
  --no_gravity_compensation Disable gravity compensation.
  --dt DT                           Simulation time-step in seconds.
```

## Acknowledgements

The controllers are borrowed from [MuJoCo Controllers](https://github.com/kevinzakka/mjctrl).

## Citation

If you find this environment useful in your research, then please consider citing:


```
@ARTICLE{10930575,
  author={Perugini, Patrizio and Lundell, Jens and Friedl, Katharina and Kragic, Danica},
  journal={IEEE Robotics and Automation Letters}, 
  title={Pushing Everything Everywhere All At Once: Probabilistic Prehensile Pushing}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/LRA.2025.3552267}}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

