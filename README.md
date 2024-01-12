# Dance_Action_Recognition

Welcome to the "Dance to the Beat with AI" project! This project uses machine learning and computer vision to capture, analyze, and evaluate dance movements in real-time. Whether you're a dance enthusiast or just curious about AI, this project provides a fun and interactive way to engage with technology.

Using OpenCV, Keras, Mediapipe, sklearn and etc.

The program can be dynamically trained to learn new dance movements and produce a new LSTM model that can recognise new actions added and recognise a sequence of actions as part of a particular dance routine.

# How it Works:

## Data Collection:

The Video class captures dance movements and creates a dataset by extracting keypoints using the MediaPipe Holistic model.
Videos are stored in the DATASET directory, and keypoints are saved in the EXTRACTION directory.

```
video = Video()
video.collect_video_keypoints()
```
## Record New Dance Actions:
The RecordVideo class allows users to record new dance actions for model training.
Videos are stored in the DATASET directory, and new keypoints are saved in the EXTRACTION directory.
```
new_actions = ["new_dance_action_1", "new_dance_action_2"]
record_video = RecordVideo(new_actions)
record_video.capture_new_dance_action()
```
## Train the Model:

The KerasModel class defines an LSTM-based neural network to classify dance actions.
The model is trained using keypoints extracted from recorded dance videos.

```
dance_model = KerasModel(action=record_video)
dance_model.set_model(load=False)  # Set load to True if you want to load pre-trained weights
```

## Create Dance Routines:

The Routine class allows users to create and store dance routines based on predefined actions.
Routines are saved in the routine_catalogue.pkl file.
```
routine = Routine(action=record_video, boolean=True)
routine.add_routine()
```
## Real-time Dance Recognition:

The go_live function uses the trained model to recognize dance actions in real-time.
It displays recognized actions on the screen and stops capturing when the "stop_capturing_movement" action is detected.

```
actions_list = ["dance_action_1", "dance_action_2", ..., "stop_capturing_movement"]
go_live(model=dance_model.get_model(), actions=actions_list, routine=routine)
```
## Tips for Users:

Ensure good lighting and keep your whole body in the frame for accurate detections.
Use "stop_capturing_movement" to end the real-time dance recognition.
Enjoy dancing to the beat with AI! 🕺💃
