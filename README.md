# Smart Mantsinen 2020-2021

## Requirements

1. Python 3.8 or higher
2. Mevea Simulation Software (Modeller and Solver)
3. Mantsinen Model
4. Playback files in .ob format

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mizolotu/SmartMantsinen
```

2. From "SmartMantsinen" directory, install required python packages (it is obviously recommended to create a new virtualenv and install everything there):
```bash
python -m pip install -r requirements.txt
```

3. Open Mevea Modeller and load Excavator Model. Go to Scripting -> Scripts, create a new script object, select "env_backend.py" from "SmartExcavator" directory as the file name in the object view. Go to I/O -> Inputs -> Input_Slew (or any other input component) and select the object just created as the script name. 

4. In terminal, navigate to the Mevea Software resources directory (default: C:\Program Files (x86)\Mevea\Resources\Python\Bin) and install numpy and requests:
```bash
python -m pip install numpy requests
```

## Preprocessing

1. From "SmartMantsinen" directory start the server script: 
```bash
python env_server.py
```
2. Convert .ob file to .csv to extract trajectory data:
```bash
python process_trajectory.py -m <path_to_mantsinen_model> -o <output_file>
```
3. (Optional) Calculate minimum and maximum data values if needed:
```bash
python standardize_data.py
```

## Behavioral clonning

## Reinforcement learning

## Learning from demonstration
