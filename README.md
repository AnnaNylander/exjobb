# How to edit the CARLA settings
1. Open the file `CarlaSettings.ini` located in `/path_to_exjobb_repo/carla_client/`
2. Here you can edit cameras etc.

# How to record data
1. Install CARLA (from source or download binary)
2. Copy the contents of directory carla_client from this repo into directory PythonClient in the CARLA folder.
3. Go to CARLA folder. (If you use the pre-compiled binaries provided by CARLA, walkers will be indluded):
`cd ~/carla`
If you instead want to run the source server (must already be manually compiled), go to
`cd ~/Repos/carla/Unreal/CarlaUE4/LinuxNoEditor`
4. To start the server, run
`./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=10 -windowed -ResX=400 -ResY=300 --carla-settings="/path_to_carla_repo/PythonClient/CarlaSettings.ini"`
6. To start the client go to PythonClient folder and run
`python3 client_1.py --frames=100 --save-path="/where/to/save/recorded/data" --carla-settings="/path_to_carla_repo/PythonClient/CarlaSettings.ini" --images-to-disk --autopilot`

# How to create data set
1. First we need to create the intentions and traffic awareness csv files.
1. Go to `/path_to_exjobb_repo/`
1. Run `python3 make_dataset.py` (with arguments) to create a dataset.
1. Then run `python3 split_dataset.py` (with arguments) to split the dataset into train, validation and test sets.

# How to visualize a frame
1. Run `python3 visualize.py --step n` where n is the frame number to visualize.
