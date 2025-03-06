# Configuration Space Computation for a 3-DOF Planar Robot

This repository contains two Python programs that demonstrate how to compute and visualize the configuration space (C-Space) of a 3-degree-of-freedom (3-DOF) planar robot, with one or multiple obstacles. Both programs use discretization of joint angles, collision detection, and A* path planning to find a safe trajectory.

## Contents

- **cspace.py**  
  Computes and visualizes the C-Space for a 3-DOF planar robot with a single obstacle.  

- **cspaceMO.py**  
  Extends the same approach to handle multiple obstacles in the workspace, providing a more complex scenario.

## Requirements
- **Python 3.x**
- **Matplotlib (for visualization)**

   ```bash
  pip install matplotlib
   ```


## How to Download

1. **Clone the repository (recommended)**  
   ```bash
   git clone https://github.com/raymundo140/autonomous-robots-using-the-configuration-space--c-space-.git
   ```

2. Navigate to the project folder
   ```bash
    cd autonomous-robots-using-the-configuration-space--c-space-
   ```
   
## How to Run

1. **Single Obstacle Scenario**
   ```bash
    python cspace.py
   ```


2. **Multiple Obstacles Scenario**

   ```bash
   python cspaceMO.py
   ```

   
