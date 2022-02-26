# Creating a panorama using traditional and deep learning approach
### Data set
![1](https://user-images.githubusercontent.com/77606010/155822494-85699084-47b6-4ef0-a27b-05a4e974acad.jpg)
![2](https://user-images.githubusercontent.com/77606010/155822495-9f9d2e8b-d297-4220-895a-60fb1a574486.jpg)
![3](https://user-images.githubusercontent.com/77606010/155822496-e675dbd1-ace3-46c9-9698-dbd9ebfc09c6.jpg)
### Final output
![p2](https://user-images.githubusercontent.com/77606010/155822514-29f98de4-f171-4888-b530-f8d0a44f9b2d.png)

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
