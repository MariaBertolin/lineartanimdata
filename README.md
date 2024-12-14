# lineartanimdata

## setup

git clone https://github.com/rtous/lineartanimdata.git
cd lineartanimdata

python3.11 -m venv myvenv
source myvenv/bin/activate

#Install SAM2 and download the checkpoints
cd sam2
pip install -e .
cd checkpoints
./download_ckpts.sh
cd ../..

#install other libraries
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

## test

1) create dir data/scenes/disney_1 and place footage.mp4 there

2) subsample

./subsample.sh $HOME/dev/lester2/data/scenes/disney_1 0

3) Speficy the segmentation settings in a file (already done for the test scene):

	  data/scenes/disney_1/scene_segmentation_settings.py


./subsample.sh $HOME/dev/lineartanimdata/data/scenes/disney_1 0

4) Segment with SAM2

	python segment_latest.py data/scenes/disney_1

5) Extract background

	python maria_foreground.py data/scenes/disney_1 

6) Color reduction

	python maria_cluster_color.py data/scenes/disney_1 

8) sketch:

	python src/ruben.py --input $HOME/dev/lineartanimdata/data/scenes/disney_1

9) cd $HOME/dev/lester2
	python maria_overlap_sketch.py data/scenes/disney_1

## test

	python all.py data/scenes/disney_1 

## test colab

change runtime type : L4 GPU