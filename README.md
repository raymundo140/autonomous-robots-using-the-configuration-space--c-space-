# Autonomous 3-DOF Planar Robot With Configuration Space Computation 

This repository contains a Python project that computes and visualizes the configuration space (C-Space) of a 3-degree-of-freedom (3-DOF) planar robot. The project discretizes the joint angles in the range \([- \pi, \pi]\) and detects collisions with obstacles present in the workspace. It then applies the A* algorithm to plan a collision-free path, and interpolates the resulting configurations to achieve a smooth animation of the robot's movement.

## Features

- **C-Space Computation:** Generates a 3D grid representing all possible robot configurations.
- **Collision Detection:** Evaluates each configuration to determine if any robot link collides with one or more obstacles.
- **Path Planning with A\*:** Uses the A* algorithm to find a valid, collision-free trajectory between an initial and a target configuration.
- **Path Interpolation:** Inserts intermediate configurations for a smoother animation.
- **Visualization and Animation:** Displays the C-Space (3D view) and a 2D simulation of the robot moving along the planned path using matplotlib.

## Requirements

- Python 3.x
- Matplotlib (install via `pip install matplotlib`)

## How to Run

Clone the repository and execute the main script:
```bash
python main.py
