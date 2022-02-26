# CMSC733 Project 1: My AutoPan

## Steps to run Phase 1 code
1. Copy the path of the Image set and paste it in the Wrapper.py's path variable (line 218). Also change the number images read on the next line.
2. Open the terminal and change path to where the code is.
3. In the terminal window, type
```
python3 Wrapper.py
```
4. The panoramic image will get saved in the folder with the name 'mypano.png'

## Steps to run Phase 2 code

### Training Supervised Model
```
python Train.py --ModelName Sup
```
### Testing Supervised Model
```
python Test.py
```
### Training Unsupervised Model
```
python Train.py --ModelName Unsup
```
### Testing Unsupervised Model
```
python Test.py --ModelPath /vulcanscratch/sonaalk/Stitching/Phase2/Checkpoints/unsupervised_small/checkpint_5.pt --ModelType Unsup
```
