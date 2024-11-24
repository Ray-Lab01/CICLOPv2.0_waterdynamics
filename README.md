# CICLOPv2.0_waterdynamics
The end to end solution for identifying and characterizing protein cavities and the water within them 

CICLOP v2.0 is a an extension of CICLOP (https://ciclop.raylab.iiitd.edu.in/), a robust Python-based molecular dynamics trajectory analysis tool tailored for identifying and characterizing protein cavities (internal cavities) and the water contained within them. This tool takes molecular dynamics simulation trajectories or a single pdb file as its input. Here, we provide a concise guide on how to use CICLOP to characterize water within protein cavities.

CICLOP is built on python3.11. The following are the dependencies required to run CICLOP
1. numpy, scipy, math, pandas, sys, os, time
2. MDAnalysis
3. tqdm
4. multiprocessing
5. shapely
6. gc
7. argparse
8. concurrent.futures
9. collections

The above dependncies can be installed seperately or using the yml file provided. And viola! You are ready to delve into the vast world of protein cavities.

A detailed turotial is also provided to help you get accustomed to the tool's usage.
