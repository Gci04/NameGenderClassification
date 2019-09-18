# Name Gender Classification

This Repository is aimed at implementing a machine learning model which classifies people given names to gender (male or female). The input names are English or Russian name. Programming language is python.

## Prerequisites

* TensorFlow >= 1.3.0
* Pandas >= 0.22.0
* Numpy >= 1.13.3
* Matplotlib >= 2.0.2
* Seaborn >= 0.7.1

## Repository directory layout

    .
    ├── Data                 # Datasets folder
    │   ├── English          # English names dataset folder
    │   └── Russian          # Russian names dataset folder
    ├── src               
    │   ├── recurrentNetwork.py        # LSTM neural network implemetation file
    │   ├── main.py                    # Main file for training and testing models
    │   ├── ClassicalNeuralNetwork.py  # file containing the implementation of Multilayer Layer Perceptron (tensorflow implemetation)
    │   └── Utils.py                   # File with implementation of data Preprocessing and some other helper methods
    │   
    ├── name_gender_classification_en.ipynb  # English names gender Classification notebook
    ├── name_gender_classification_ru.ipynb  # Russian names gender Classification notebook
    └── README.md


## Data Description

In this repository Russian Name gender [dataset from Kaggle](https://www.kaggle.com/rai220/russian-cyrillic-names-and-sex) is used. The original data from kaggle has 4 features which are [Surname, Name, Middlename , gender] and for this repository only Name and Gender are used.
```
    English                   Russian
+ - - - - +- - - - +     + - - - - - - - - - +
|   Name  | Gender |     |   Name  | Gender  |
+ - - - - +- - - - +     + - - - - + - - - - +
| Terrone |    M   |     | ДМИТРИЙ |   M     |
| Annaley |    F   |     | ЕЛЕНА   |   Ж     |
| Alajha  |    F   |     | РАВИЛЬ  |   M     |
|  ...    |  ..... |     |  ...    | ......  |
+ - - - - +- - - - +     + - - - - + - - - - +
```

<!-- ## Data Preprocessing -->

## Classification Models
* Multilayer Perceptron - implemented in TensorFlow
* Recurrent Neural Network - Long short-term memory (LSTM)

## Models Evaluations and Report

## References
