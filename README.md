<div style="text-align:center">
    <img src="/imgs/cfnow_hq_logo.gif" alt="image" width="100%"/>
</div>


## CFNOW - CounterFactual Nearest Optimal Wololo
![unittests](https://github.com/rmazzine/CFNOW/actions/workflows/unittests.yaml/badge.svg)
[![codecov](https://codecov.io/gh/rmazzine/CFNOW/graph/badge.svg?token=4NHY0V9CN9)](https://codecov.io/gh/rmazzine/CFNOW)


### Description

> TL;DR: You just need a `dataset point` and a `model prediction function`. CFNOW will find the closest point with a different class.

The simplest way to generate counterfactuals for any tabular dataset and model.

This package finds an optimal point (closer to the  input dataset point), which the classification is different from the original classification (i.e. "flips" the classification of the original input by minimally changin it).

## How to install
    
```bash
    pip install cfnow
```

## Table of Contents

- [Minimal example](#minimal-example)
- [Counterfactual Charts](#showing-the-counterfactuals-graphically)
- [I have: binary categorical features!](#i-have-binary-categorical-features)
- [I have: one-hot encoded features!](#i-have-one-hot-encoded-features)
- [I have: binary and OHE features!](#i-have-a-mix-of-binary-categorical-and-one-hot-encoded-features)
- [Image Counterfactuals](#image-counterfactuals)
- [Text Counterfactuals](#text-counterfactuals)
- [How to cite](#how-to-cite)

## Requirements

- Python >= 3.8

### Minimal example:
```python
from cfnow import find_tabular
import sklearn.datasets
import sklearn.ensemble
import pandas as pd

# Generating a sample model
X, y = sklearn.datasets.load_iris(return_X_y=True)
model = sklearn.ensemble.RandomForestClassifier()
model.fit(X, y)

# Selecting a random point
x = X[0]

# Here we can see the original class
print(f"Factual: {x}\nFactual class: {model.predict_proba([x])}")

# Then, we use CFNOW to generate the minimum modification to change the classification
cf_obj = find_tabular(
    factual=pd.Series(x),
    model_predict_proba=model.predict_proba,
    limit_seconds=10)

# Here we can see the new class
print(f"CF: {cf_obj.cfs[0]}\nCF class: {model.predict_proba([cf_obj.cfs[0]])}")
```

### Showing the Counterfactuals graphically
This package is integrated with [CounterPlots](https://github.com/ADMAntwerp/CounterPlots), that allows you to graphically represent your counterfactual explanations!

You can simply generate Greedy, CounterShapley, and Constellation charts for a given CF with:
#### Greedy
```python
# Get the counterplots for the first CF and save as greedy.png
cf_obj.generate_counterplots(0).greedy('greedy.png')
```
#### Output Example
![image](/imgs/greedy_ex.png)
#### CounterShapley
```python
# Get the counterplots for the first CF and save as greedy.png
cf_obj.generate_counterplots(0).countershapley('countershapley.png')
```
![image](/imgs/countershapley_ex.png)
#### Constellation
```python
# Get the counterplots for the first CF and save as greedy.png
cf_obj.generate_counterplots(0).constellation('constellation.png')
```
![image](/imgs/const_ex.png)

### Improving your results
The minimal example above considers all features as numerical continuous, however, some datasets can have categorical (binary or one-hot encoded) features. CFNOW can handle these data types in a simple way as demonstrated below:

### I have binary categorical features!
#### 1 - Prepare the dataset
```python
import pandas as pd
import numpy as np
import sklearn.ensemble

# Generate data with 5 binary categorical features and 3 continuous numerical features
X = np.hstack((
    np.random.randint(0, 2, size=(1000, 5)),
    np.random.rand(1000, 3) * 100
))

# Random binary target variable
y = np.random.randint(0, 2, 1000)

# Train RandomForestClassifier
model = sklearn.ensemble.RandomForestClassifier().fit(X, y)

# Display the original class for a random test sample
x = X[0]
print(f"Factual: {x}\nFactual class: {model.predict_proba([x])}")
```

#### 2 - Find the CF
```python
from cfnow import find_tabular
# Then, we use CFNOW to generate the minimum modification to change the classification
cf_obj = find_tabular(
    factual=pd.Series(x),
    feat_types={i: 'cat' if i < 5 else 'cont' for i in range(8)},
    model_predict_proba=model.predict_proba,
    limit_seconds=10)

# Here we can see the new class
print(f"CF: {cf_obj.cfs[0]}\nCF class: {model.predict_proba([cf_obj.cfs[0]])}")



```

### I have one-hot encoded features!
#### 1 - Prepare the dataset
```python
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sklearn.ensemble
warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names")


# Generate data
X = np.hstack((np.random.randint(0, 10, size=(1000, 5)), np.random.rand(1000, 3) * 100))

# One-hot encode the first 5 categorical columns

# !!!IMPORTANT!!! The naming of OHE encoding features columns MUST follow feature_value format.
# Therefore, for a feature called color with value equal to red or blue, the OHE encoding columns
# must be named color_red and color_blue. Otherwise, the CF will not be able to find the correct
# columns to modify.
encoder = OneHotEncoder(sparse=False)
X_cat_encoded = encoder.fit_transform(X[:, :5])
names = encoder.get_feature_names_out(['cat1', 'cat2', 'cat3', 'cat4', 'cat5'])

# Combine and convert to DataFrame
df = pd.DataFrame(np.hstack((X_cat_encoded, X[:, 5:])), columns=list(names) + ['num1', 'num2', 'num3'])

# Random binary target variable
y = np.random.randint(0, 2, 1000)

# Train RandomForestClassifier
model = sklearn.ensemble.RandomForestClassifier().fit(df, y)

# Display the original class for a random test sample
x = df.iloc[0]
print(f"Factual: {x.tolist()}\nFactual class: {model.predict_proba([x])}")
```

#### 2 - Find the CF
```python
from cfnow import find_tabular
# Then, we use CFNOW to generate the minimum modification to change the classification
cf_obj = find_tabular(
    factual=x,
    feat_types={c: 'cat' if 'cat' in c else 'cont' for c in df.columns},
    has_ohe=True,
    model_predict_proba=model.predict_proba,
    limit_seconds=10)

# Here we can see the new class
print(f"CF: {cf_obj.cfs[0]}\nCF class: {model.predict_proba([cf_obj.cfs[0]])}")
```

### I have one-hot and binary categorical features!
#### 1 - Prepare the dataset
```python
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sklearn.ensemble
warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names")


# Generate data
X = np.hstack((np.random.randint(0, 10, size=(1000, 5)), np.random.rand(1000, 3) * 100))

# One-hot encode the first 5 categorical columns

# !!!IMPORTANT!!! The naming of OHE encoding features columns MUST follow feature_value format.
# Therefore, for a feature called color with value equal to red or blue, the OHE encoding columns
# must be named color_red and color_blue. Otherwise, the CF will not be able to find the correct
# columns to modify. For binary, it is just sufficient to name refer the column as cat.
encoder = OneHotEncoder(sparse=False)
X_cat_encoded = encoder.fit_transform(X[:, :5])
names = encoder.get_feature_names_out(['cat1', 'cat2', 'cat3', 'cat4', 'cat5'])

# Combine and convert to DataFrame
df = pd.DataFrame(np.hstack((X_cat_encoded, X[:, 5:])), columns=list(names) + ['num1', 'num2', 'num3'])

# Random binary target variable
y = np.random.randint(0, 2, 1000)

# Train RandomForestClassifier
model = sklearn.ensemble.RandomForestClassifier().fit(df, y)

# Display the original class for a random test sample
x = df.iloc[0]
print(f"Factual: {x.tolist()}\nFactual class: {model.predict_proba([x])}")
```

#### 2 - Find the CF
```python
from cfnow import find_tabular
# Then, we use CFNOW to generate the minimum modification to change the classification
cf_obj = find_tabular(
    factual=x,
    feat_types={c: 'cat' if 'cat' in c else 'cont' for c in df.columns},
    has_ohe=True,
    model_predict_proba=model.predict_proba,
    limit_seconds=10)

# Here we can see the new class
print(f"CF: {cf_obj.cfs[0]}\nCF class: {model.predict_proba([cf_obj.cfs[0]])}")
```

## Image Counterfactuals

To generate image counterfactuals, we will use some additional packages to help us.

Below we show the process to generate a counterfactual for the image below:

<div style="text-align:center">
    <img src="/imgs/example_factual_img_daisy.png" alt="image" width="50%"/>
</div>

#### factual class: daisy

<div style="text-align:center">
    <img src="/imgs/example_cf_img_daisy.png" alt="image" width="50%"/>
</div>

#### cf class: bee

### 1 - Install the packages
```bash
pip install torch torchvision Pillow requests
```

We also recommend to run this experiment in Jupyter Notebook, as it will be easier to visualize the results.

Most of the code for this example is to load the pre-trained model and the image. The CFNOW part is very simple.

### 2 - Loading the pre-trained model and building image classifier
```python
import requests
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fetch an image from the web
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia2.jpg/320px-Sunflower_from_Silesia2.jpg"
response = requests.get(image_url, stream=True)
image = np.array(Image.open(response.raw))

def predict(images):
    if len(np.shape(images)) == 4:
        # Convert the list of numpy arrays to a batch of tensors
        input_images = torch.stack([transform(Image.fromarray(image.astype('uint8'))) for image in images])
    elif len(np.shape(images)) == 3:
        input_images = transform(Image.fromarray(images.astype('uint8')))
    else:
        raise ValueError("The input must be a list of images or a single image.")
    
    # Check if a GPU is available and if not, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_images = input_images.to(device)
    model.to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(input_images)
    
    # Return an array of prediction scores for each image
    return torch.asarray(outputs).cpu().numpy()

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
def predict_label(outputs):
    # Load the labels used by the pre-trained model
    labels = requests.get(LABELS_URL).json()
    
    # Get the predicted labels
    predicted_idxs = [np.argmax(od) for od in outputs]
    predicted_labels = [labels[idx.item()] for idx in predicted_idxs]
    
    return predicted_labels

# Check the prediction for the image
predicted_label = predict([np.array(image)])
print("Predicted labels:", predict_label(predicted_label))
```

To find the CF you just need to:

### 3 - Find the CF
```python
from cfnow import find_image

cf_img = find_image(img=image, model_predict=predict)

cf_img_hl = cf_img.cfs[0]
print("Predicted labels:", predict_label(predict([cf_img_hl])))

# Show the CF image
Image.fromarray(cf_img_hl.astype('uint8'))
```

## Text Counterfactuals

You can also generate counterfactuals for embedding models.

For this example, you will need to install the following packages:


### 1 - Install the packages
```bash
pip install transformers
```

### 2 - Load the pre-trained model
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline

import numpy as np

# Load pre-trained model and tokenizer for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Define the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Define a simple dataset
text_factual = "I liked this movie because it was funny but my friends did not like it because it was too long and boring."

result = sentiment_analysis(text_factual)
print(f"{text_factual}: {result[0]['label']} (confidence: {result[0]['score']:.2f})")
        
def pred_score_text(list_text):
    if type(list_text) == str:
        sa_pred = sentiment_analysis(list_text)[0]
        sa_score = sa_pred['score']
        sa_label = sa_pred['label']
        return sa_score if sa_label == "POSITIVE" else 1.0 - sa_score
    return np.array([sa["score"] if sa["label"] == "POSITIVE" else 1.0 - sa["score"] for sa in sentiment_analysis(list_text)])
```

In the code above, we can see the factual text has a NEGATIVE classification with a high confidence.
```text
I liked this movie because it was funny but my friends did not like it because it was too long and boring.: NEGATIVE (confidence: 0.98)
```

### 3 - Find the CF
```python
from cfnow import find_text
cf_text = find_text(text_input=text_factual, textual_classifier=pred_score_text)
result_cf = sentiment_analysis(cf_text.cfs[0])
print(f"CF: {cf_text.cfs[0]}: {result_cf[0]['label']} (confidence: {result_cf[0]['score']:.2f})")
```

The CF text has a POSITIVE classification with a high confidence.
```text
CF: I liked this movie because it was funny  my friends did not like it because it was too long and boring.: POSITIVE (confidence: 0.93)
```

In the example above, we can see how minimal the difference is to a change in the classification. Where just one word (`but`) was removed from the text.

## How to cite
If you use CFNOW in your research, please cite the following paper:
```
@article{DEOLIVEIRA2023,
title = {A model-agnostic and data-independent tabu search algorithm to generate counterfactuals for tabular, image, and text data},
journal = {European Journal of Operational Research},
year = {2023},
issn = {0377-2217},
doi = {https://doi.org/10.1016/j.ejor.2023.08.031},
url = {https://www.sciencedirect.com/science/article/pii/S0377221723006598},
author = {Raphael Mazzine Barbosa {de Oliveira} and Kenneth Sörensen and David Martens},
}
```