# Investigating the Viability of Generating Trading Strategies with Multi-Objective Evolutionary Algorithms
## Github Repository link

Contains external libraries in a virtual environment. This virtual environent was too large to submit. Git hub link:

https://github.com/BenTrotter/ECM3401_BenTrotter.git

## Demonstration video link

The video zipped to the submission has been compressed so the quality of this video may be reduced making it harder to read the slides. Therefore it is recommened to view the video through the youtube link below as it will be in a higher quality:

https://youtu.be/b7TvS3BT2ME

# ECM3401 - Source Code
## General Overview

This program investigates using NSGA-II with strongly typed genetic programming to find successful trading strategies.

## Running the program
### Clone the repository and start the virtual environment

The virtual environment is named evolution_env and contains the relevant libraries and dependancies that are needed to run the program successfully. The following code can demonstrate how to start the environment.

```shell
$ cd evolution_env
$ source bin/activate
$ cd ..
```

### Configure globals.py

Depending on the tests being carried out, the parameters for the evolutionary process need to be changed. This can be achieved by opening and editing gloabls.py. This file holds all the global variables in one place to allow for consistency across the files.

### Run evolution.py

The final step is to run the program. Use the command below to run the program using parallel processing.

```shell
$ python3 -m scoop evolution.py

