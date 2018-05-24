In the following guides it is assumed that this repo (exjobb) is located in the home folder.

# How to edit the CARLA settings
1. Open the file `CarlaSettings.ini` located in `~/exjobb/carla/PythonClient`
2. Here you can edit cameras etc.

# How to record data
To run the source server (must already be manually compiled), go to
`cd ~/exjobb/carla/Unreal/CarlaUE4/LinuxNoEditor`
2. To start the server, run
`./CarlaUE4.sh /Game/Maps/Town01 -carla-server -benchmark -fps=10 -windowed -ResX=400 -ResY=300`
3. To start the client go to PythonClient folder and run
`sudo python3 client_record_data.py -f 100 -s /where/to/save/recorded/data/ -c CarlaSettings.ini -i -a -n name_of_recording`
4. You can edit and run `bash record.sh` in the exjobb root folder to record consecutive sessions.

# How to create data set
1. First we need to create the intentions and traffic awareness csv files.
1. Go to `~/exjobb/`
1. Run `python3 make_dataset.py` (with arguments) to create a dataset.
1. Then run `python3 split_dataset.py` (with arguments) to split the dataset into train, validation and test sets.

# How to visualize a frame
1. Run `python3 visualize.py --step n` where n is the frame number to visualize.
