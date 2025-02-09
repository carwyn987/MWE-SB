# MWE-SB: A Minimum Working Example Stable Baseline for Reinforcement Learning Algorithms

When you're starting to learn about RL algorithms, you don't want to interact with mature codebases. You want to see the start to end, the environment setup, network setup, and learning algorithm right out in front of you. You don't want to have to switch files often and deal with the fragmentation and class proliferation that is all-too common with object oriented programming.

This repo is just that - the basic algorithms, implemented for a particular environment. Everything needed is contained within one file, and a series of standardized imports. Just set up a venv with the requirements.txt file, and run. 

As I've implemented more algorithms, I have became interested in comparing performance and ablation testing. Therefore I have also implemented a  plotter and return saving function in the `plotter/` folder. This is integrated with each self contained file, HOWEVER, it is completely optional. Removing this function will not break anything, and it's calls make up a tiny surface area in each file.