# A brief overview of TensorFlow

## Installation
### Check Python version  (should be 2.7+)
	python -V

### Install pip for Python 2.7
	wget https://bootstrap.pypa.io/get-pip.py
	python get-pip.py

### Install other dependencies
	pip install -r requirements.txt
	
### Install TensorFlow
Currently, TensorFlow not included in the pip repository, we have to install it manually.

	pip install --upgrade https://storage.googleapis.com/tensorflow/mac/	tensorflow-0.6.0-py2-none-any.whl

## Run the demo application
	python mnist.py
	
It will take about one hour and a half to train the dataset.
		
