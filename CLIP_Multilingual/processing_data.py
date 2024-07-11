# Importing os, numpy and pandas for data manipulation
import os
import numpy as np
import pandas as pd
# For data visualization, we will use matplotlib, wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import clip
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import json
from tqdm.notebook import tqdm
from PIL import Image
import torch

def read_dataframe(path):
    """
    Reads the JSON file and returns a dataframe with the required columns (image, question, answers, answer_type, answerable)

    Parameters:
        path (str): Path to the JSON file

    Returns:
        df (pandas.DataFrame): Dataframe with the required columns
    """
    #df = pd.read_json(path)
    df = pd.read_csv(path)
    df['answers_trad']=df['answers_trad'].apply(lambda x:eval(x))
    #agrego este renombre asi pued usar los textos traducidos al espaol que los tengo en el mismo archivo
    df.rename(columns={'question': 'question_orig','answers':'answers_orig','question_trad': 'question','answers_trad': 'answers'},inplace=True)
    
    df = df[['image', 'question', 'answers', 'answer_type', 'answerable']]
    return df

def split_train_test(dataframe, test_size = 0.05):
    """
    Splits the dataframe into train and test sets

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be split

    Returns:
        train (pandas.DataFrame): Train set
        test (pandas.DataFrame): Test set
    """
    train, test = train_test_split(dataframe, test_size=test_size, random_state=42, stratify=dataframe[['answer_type', 'answerable']])
    return train, test

def plot_histogram(dataframe, column):
    """
    Plots the histogram of the given column

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be plotted
        column (str): Column to be plotted

    Returns:
        None
    """
    plt.hist(dataframe[column])
    plt.title(column)
    plt.show()

def plot_pie(dataframe, column):
    """
    Plots the pie chart of the given column

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be plotted
        column (str): Column to be plotted

    Returns:
        None
    """
    value_counts = dataframe[column].value_counts()
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
    plt.title(column)
    plt.show()

def plot_wordcloud(dataframe, column):
    """
    Plots the wordcloud of the given column

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be plotted
        column (str): Column to be plotted

    Returns:
        None
    """
    text = " ".join([word for word in dataframe[column]])
    #print('text: ', text)
    wordcloud = WordCloud(width = 500, height = 500,
                    background_color ='white',
                    min_font_size = 10).generate(text)

    plt.figure(figsize = (6, 6), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

def explore_dataframe(dataframe):
    """
    Explores the dataframe (EDA) by plotting the pie charts, histograms and wordclouds of the columns

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be explored

    Returns:
        None
    """
    plot_pie(dataframe, 'answer_type')
    #plot_pie(dataframe, 'answerable')
    #plot_histogram(dataframe, 'answerable')
    plot_wordcloud(dataframe, 'question')

def get_number_of_distinct_answers(dataframe):
    """
    Returns the number of distinct answers in the dataframe

    Parameters:
        dataframe (pandas.DataFrame): Dataframe to be explored

    Returns:
        len(unique_answers_set) (int): Number of distinct answers in the dataframe
    """
    unique_answers_set = set()
    for row in dataframe['answers']:
        for answer_map in row:
            #print('answer_map]: ', answer_map)
            #unique_answers_set.add(answer_map['answer']) cambie esta linea
            unique_answers_set.add(answer_map)
    return len(unique_answers_set)

def process_images(dataframe, image_path, clip_model, device):
    """
    Processes the images in the dataframe and returns the image features

    Parameters:
        dataframe (pandas.DataFrame): Dataframe containing the images
        image_path (str): Path to the input images
        clip_model (clip.model.CLIP): CLIP model
        preprocessor (clip.model.Preprocess): Preprocessor for the CLIP model
        device (torch.device): Device to be used for processing

    Returns:
        images (list): List of image features
    """
    images = []
    for _, row in tqdm(dataframe.iterrows()):
        full_path = image_path + row['image'] #+ '.jpg'
        image = Image.open(full_path)
        #image = preprocessor(image).unsqueeze(0).to(device)
        #image_features = clip_model.encode_image(image)
        #image_features = torch.flatten(image_features, start_dim=1)
        image_features = clip_model.encode(image,convert_to_tensor=True)
        image_features = image_features.unsqueeze(0)

        images.append(image_features)
    print('image_features size: ',images[0].size())
    return images

def process_images_orig(dataframe, image_path, clip_model, preprocessor, device):
    """
    Processes the images in the dataframe and returns the image features

    Parameters:
        dataframe (pandas.DataFrame): Dataframe containing the images
        image_path (str): Path to the input images
        clip_model (clip.model.CLIP): CLIP model
        preprocessor (clip.model.Preprocess): Preprocessor for the CLIP model
        device (torch.device): Device to be used for processing

    Returns:
        images (list): List of image features
    """
    images = []
    for _, row in tqdm(dataframe.iterrows()):
        full_path = image_path + row['image'] #+ '.jpg'
        image = Image.open(full_path)
        image = preprocessor(image).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image)
        image_features = torch.flatten(image_features, start_dim=1)
        images.append(image_features)
    return images

def process_questions(dataframe, clip_model,device):
    """
    Processes the questions in the dataframe and returns the question features

    Parameters:
        dataframe (pandas.DataFrame): Dataframe containing the questions
        clip_model (clip.model.CLIP): CLIP model
        device (torch.device): Device to be used for processing

    Returns:
        questions (list): List of question features
    """
    questions = []
    for _, row in tqdm(dataframe.iterrows()):
        question = row['question']
        #question = clip.truncate(question, 77) #agregp esto porque en la version traducida me aparecieron preguntas con mas de 77 tockens q es lo permitido por clip
        #question =  clip.tokenize(question).to(device)
        #question =  clip.tokenize(question,truncate= True).to(device)#agregp esto porque en la version traducida me aparecieron preguntas con mas de 77 tockens q es lo permitido por clip
        #text_features = clip_model.encode_text(question).float()
        #text_features = clip_model.encode(question).float()
        #text_features = torch.flatten(text_features, start_dim=1)
        text_features = clip_model.encode([question], convert_to_tensor=True)
        questions.append(text_features)
    print('text_features size: ', text_features[0].size())
    return questions