import ollama
import pandas as pd

def llm_pull(model):
    """
    Pulls the model from the Ollama API

    Parameters:
    model (str): The name of the model to pull
    """
    ollama.pull(model)

def llm_call(messages, model='mistral'):
    """
    Calls the model with the given messages

    Parameters:
    messages (list): A list of dictionaries containing the role and content of each message
    model (str): The name of the model to call
    """

    # Check if the model is pulled
    if model not in ollama.models():
        llm_pull(model)

    response = ollama.chat(
        model = model,
        messages = messages,
        stream = False
    )
    return response['message']['content']

def create_prompt_content(title, description):
    """
    Create a content object from the given title and description

    Parameters:
    title (str): The title of the video
    description (str): The description of the video
    """

    prompt = {'Using the following title and description, predict whether the video is an apology video: \n Title: ' + title + '\n Description: ' + description + '\n Answer by True or False.'}

    return prompt

def apology_video(title, description):
    """
    Check whether the model identifies a video as an apology video from the given title and description

    Parameters:
    title (str): The title of the video
    description (str): The description of the video
    """

    messages = [
        {
            'role': 'user',
            'content': create_prompt_content(title, description)
        },
    ]

    response = llm_call(messages)
    return response['content']

def apology_in_week(channel_id, week_index, n=None):
    """
    Check whether the model finds an apology video in the n next videos of a channel, starting from a given week
    If n is None, the model checks all the videos from the week_index

    Parameters:
    channel_id (str): The id of the channel to check
    week_index (int): The index of the week to check
    n (int): The number of videos to check
    """

    apology_found = False

    for i in range(n):
        response = apology_video()
        if response == True:
            apology_found = True
            break
    
    return apology_found


def load_video_metadata(usecols):
    """
    Load the video metadata from the file and return the first n_first_rows
    Parameters:
    filename (str): the name of the file to load
    n_first_rows (int): the number of rows to load
    Return:
    df (pd.DataFrame): the first n_first_rows of the file
    """
    
    df_video_metadata = pd.read_csv(f'./../../data/video_metadata.csv', usecols=usecols)
    return df_video_metadata