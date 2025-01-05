# lineartanimdata

## Setup

git clone https://github.com/MariaBertolin/lineartanimdata.git
cd lineartanimdata

python3.11 -m venv myvenv
source myvenv/bin/activate

#Install SAM2 and download the checkpoints

cd sam2
pip install -e .
cd checkpoints
./download_ckpts.sh
cd ../..

#Install other libraries

pip install matplotlib==3.9.2
pip install opencv-python==3.4.17.61
pip install numpy==1.26.0

#sketchKeras-pytorch

pip install gdown
mkdir weights
cd weights
gdown 1Zo88NmWoAitO7DnyBrRhKXPcHyMAZS97
cd ..
pip install "numpy<1.24" 

## Test

1) Create dir data/scenes/SCENENAME and place footage.mp4 there (already done for the test scene)

2) Subsample

	./subsample.sh $HOME/dev/lineartanimdata/data/scenes/test 0

3) Speficy the segmentation settings in a file (already done for the test scene):

	  data/scenes/test/scene_segmentation_settings.py

4) Segment with SAM2

	python segment_latest.py data/scenes/test

5) Extract background

	python step2_foreground.py data/scenes/test 

6) Color reduction

	python step3_cluster_color.py data/scenes/test 

8) Sketch (Simple sketch = maria.py / SketchKeras sketch = ruben.py):

	python src/maria.py --input $HOME/dev/lineartanimdata/data/scenes/test

9) cd $HOME/dev/lester2
	
	python step5_overlap_sketch.py data/scenes/test

## Test

	python all.py data/scenes/test 

## Test colab
-> Use the notebook lineartanimdata.ipynb

-> Change runtime type : L4 GPU
