This README will act as a guide for running the scripts specificly related to annotations and inspections for our DARTS project.

--------------------------
createMasks.py
This script will be used to both visually confirm your annotations were done correctly and that you know how to create a python virtual environment.

Python Virtual Environments (Conda and Venv). You need to know how to do both as Case HPC does not have Conda available as an option (in some circumstances) although Conda is typically considered to be easier. Also conda has some tools that Venv does not.

Step 1. Install python
Step 2. Choose your variation (see variations below)
Step 3. Activate your environment and confirm it's activated by looking at the name on the left side of the terminal.
Step 4. Install pip with:
        conda install pip
Step 5. Normally, a package will include a file called 'requirements.txt'. If this file is included all of the package's dependencies can be installed with:
        pip install -r requirements.txt
However, I am very intentionally not including one because it is an essential skill to know how to read the output errors and figure out which dependencies need to be installed. The point of this excerise is to prepare you for the very real situation of what is commonly referred to as 'dependency hell'. This package does not have very many depedencies to figure out so it shouldn't take longer than a few minutes if you've done it before. If you haven't, now is a great time to learn by doing!
Step 6. Navigate through your directories (these are bash commands) to the main folder that 
contains the file 'createMasks.py'
        cd <folder-name> # moves you into a directory
        ls #Shows all of the files in your current directory
        cd .. # moves you back to the directory you were just in
Step 7. Open the script. Change the filepath variables to the filepath locations of YOUR folders in the Week1 folder. If you haven't created the output folders (mask and overlay) it should create it for itself. Don't forget to save.
Step 8. Run the script:
        python createMasks.py


Variation 1 Conda
Step 1. Download Conda (https://anaconda.org/anaconda/conda)
Step 2. Go to a terminal and confirm conda is working with:
        conda --version
Step 3. Create a conda environment with by following the documentation (https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)
        conda create --name <my-env>
        proceed ([y]/n)?
Step 5. Activate the environment with:
        conda env list
        conda activate <my-env>



Variation 2 Venv
Step 1. Create Venv environment (no need to install, comes preinstalled with python)
        python -m venv C:\path\to\new\virtual\environment
Step 2. Activate the environment (https://www.geeksforgeeks.org/create-virtual-environment-using-venv-python/):
        source <venv-name>/bin/activate # for linux