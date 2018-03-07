# How to record data
1. Install CARLA
2. Copy the contents of directory carla_client from this repo into Python_client in the CARLA repo.
3. Start the CARLA server with the following command:

`./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=10 -windowed -ResX=800 -ResY=600 --carla-settings="/path_to_carla_repo/Python_client/CarlaSettings.ini"`

4. Edit `client_1.py` to set the path of saved data and the number of frames to run.
5. cd to Python_client and run 
`python3 client_1.py --carla-settings="/path_to_carla_repo/Python_client/CarlaSettings.ini" --images-to-disk --autopilot`

# How to create data set
1. Run `python3 make_dataset.py` and the input and output will be created in the directory specified in the same file.

# How to visualize a frame
1. Run `python3 visualize.py --step n` where n is the frame number to visualize.

