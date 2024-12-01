# Empirical Study: Road Sign Classification
Official Code for CSE 881 Fall 2024 Final Project: "Empirical Study: Road Sign Classification" Bao Hoang and Tanawan Premsri.

## Overview
 The rapid advancement of autonomous driving technology has highlighted the critical need for robust road sign detection systems. This project leverages Computer Vision and Deep Learning Networks to develop and evaluate models for detecting and classifying road signs. Diverse datasets, including images from Google Images, Google Shopping, and Kaggle, are collected and used to train various advanced architectures. Performance is assessed by splitting the data into training (80\%) and test (20\%) sets, with additional evaluations conducted on different data sources, such as Kaggle and Google. Furthermore, incorrect model predictions are analyzed to identify underlying causes and improve model accuracy. Our codes are provided in [https://github.com/hoangcaobao/CSE881-Fall2024-FinalProject](https://github.com/hoangcaobao/CSE881-Fall2024-FinalProject). Our deployed website URL is [https://appdeploytest-wv9jmdfc27xqm9crhs3uad.streamlit.app/](https://appdeploytest-wv9jmdfc27xqm9crhs3uad.streamlit.app/).

## Package dependencies
Run ```pip install -r requirements.txt```

## Data preparation
### Google Data Scraping
To scrap images from Google, run ```python3 scrape.py```

### Kaggle Dataset
Please download images from the Kaggle website ([Road Sign Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection/data)) and place them in a folder named kaggle.

### Generate Synthetic Data. 
```python generate_syn_image.py --cuda 0 --num_repeat 200 --num_inference_step 80```

### Generate Meta Data
Finally, after gathering the Google Dataset, Kaggle Dataset, and Synthetic Dataset, we will generate a metadata file using the image preprocessing function and split the data into training and testing portions.
First, navigate to the data_preprocessing directory by running ```cd data_preprocessing```, then run the following command ```python create_image_dataset.py```

## Demos
Here we provide several demos of results in the project report.

### 1. ResNet-50:
+ Train on Google dataset: ```python train.py --model RESNET --train google```
+ Train on Kaggle dataset: ```python train.py --model RESNET --train kaggle```
+ Train on both: ```python train.py --model RESNET --train all```

### 2. VGG-16:
+ Train on Google dataset: ```python train.py --model VGG --train google```
+ Train on Kaggle dataset: ```python train.py --model VGG --train kaggle```
+ Train on both: ```python train.py --model VGG --train all```

### 3. Inference CLIP:
```python inference_CLIP.py --cuda 0 --train_set all``` to generate csv results, then run ```python check_acc.py --result_path CLIP_result.csv``` to check accuracy.

### 4. Train a classifier on CLIP's embedding:
+ Train on Google dataset: ```python train_CLIP_classifier.py --epoch 40 --train_set google``` 
+ Train on Kaggle dataset: ```python train_CLIP_classifier.py --epoch 40 --train_set google``` 
+ Train on both: ```python train_CLIP_classifier.py --epoch 40 --train_set all``` 

### 5. Inference Llava:
Run ```python inference_Llava.py --data_path metadata_test``` to generate csv results, then run ```python check_acc.py --result_path result_llava_next_meta_test.csv``` to check accuracy.

### 6. Llava + BLIP:
+ Firstly, we need to generation caption: ```python generate_caption.py --data_path metadata_test```
+ Then, run: ```python inference_Llava.py --use_BLIP T --data_path metadata_test``` to generate csv results.
+ Finally, run ```python check_acc.py  --result_path result_llava_next_meta_test.csv``` to check accuracy.

## Run app
We can run the app by ```streamlit run app.py``` to run locally. We also host the app through this GitHub Repo: [https://github.com/hoangcaobao/StreamlitDeploy](https://github.com/hoangcaobao/StreamlitDeploy).

## Find Project Report and Presentation Slides on Report.pdf and Presentation.pdf
  
 
