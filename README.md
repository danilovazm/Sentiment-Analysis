# Sentiment-Analysis

## Status

On-Hold

## About

The main goal of this project is to classify a movie review into a posivite or negative one.

### Dataset
For this project this [public dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from kaggle was used.

### Implementation
The code was implemented in python using pytorch, pandas, numpy, spacy, scikit-learn, matplotlib and seaborn libraries. The dataset was preprocessed using spacy function, tokenization, lemmatization and vectorization were applyied to the reviews and the data was splitted in train/test according to the 80/20 ratio. The model used has the following architecture: Lstm block, with 4 layers and a hidden layer with 128 neurons, followed by a fully-conected layer and a sigmoid.

### Training
The training of the model was perform through 50 epochs using the Adam optimizer with a learning rate of ``` 0.0001 ``` and the objective function was the BCE using a batch of 1 due to hardware limitation. 

### Summary
`Main.py`: file that calls the training and testing of the model based on users choice, prints and plots the model's result.

`PreProcess.py`: file where the preprocess of the reviews is implemented.

`Loader.py`: file where de dataloader is implemented.

`Network.py`: file that contains the definition of the model architecture.

`Train.py`: file where the training routine is implemented.

`Test.py`: file where the testing routine is implemented.

## How To
The python version used in this project was:
```
    Python 3.7.3
```

First you will need to install the following dependencies:
``` 
    pip install torch torchvision torchaudio
    pip install pandas
    pip install numpy
    pip install spacy
    pip install scikit-learn
    pip install seaborn
    pip install matplotlib
```
The command to install pytorch differ according to your setup, check the [docs](https://pytorch.org/get-started/locally/) to see the version that would fit you better

To download the model used to realize the vectorization execute the following command:
``` 
    python -m spacy download en_core_web_md
```

Once the dependencies were installed to run the code go to the root of this repository and execute the command below:
``` 
    python Main.py
```

Also, when running the Main.py file, is possible to fit some methods' parameters. To do so choose at least one of the following flags defining the desired value:

`--epochs`: define the number of epochs you want to perform the training, default 50.

`--lr`: Choose the learning rate you want to use, default `0.0001`.

`--batch_size`: Define the batch size you wanna use, default 1.

`--reviews`: Choose how many reviews you wanna use to execute the training and testing., default 10. (You should use a much higher number of reviews to get a good result) 

`--split`: Define the split ratio between train and test, default 0.8.

## Results

The results are not satisfactory yet. The best result was `58%` of accuracy using 10000 reviews, due to time limitation was not possibly use the entire dataset

## Next steps

- Use another library to do the preprocessing, `nltk` probably.
- Explore others models.
