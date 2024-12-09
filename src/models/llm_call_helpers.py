import ollama
import pandas as pd

GLOBAL_CONTEXT = "You are analyzing YouTube videos to determine how creators handle a decline in their channel popularity. \n The video is analyzed based on its title and description. Your task is to answer the given specific question"

def llm_pull(model):
    """
    Pulls the model from the Ollama API

    Parameters:
    model (str): The name of the model to pull
    """
    ollama.pull(model)

def query_ollama(prompt, model="mistral"):
    """
    Query the Ollama-hosted model locally

    Parameters:
    prompt (str): The prompt to query the model with
    model (str): The name of the model to query

    Returns:
    str: The response from the model

    Raises:
    RuntimeError: If an error occurs while querying the model
    """

        # Check if the model is pulled
    if model not in ollama.list():
        llm_pull(model)

    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response.get('response')
    except Exception as e:
        raise RuntimeError(f"Error querying the local model: {e}")

def create_prompt(context, title, description, question):
    """
    Construct a prompt with context, video details, and a specific question.

    Parameters:
    context (str): The context of the question
    title (str): The title of the video
    description (str): The description of the video
    question (str): The question to ask

    Returns:
    str: The constructed prompt
    """
    return (
        f"Context:\n"
        f"{context}\n\n"
        f"Video Details:\n"
        f"Title: {title}\n"
        f"Description: {description}\n\n"
        f"Question:\n"
        f"{question}\n\n"
        f"Answer with 'True' or 'False' only."
    )

def apology_video(title, description):
    """
    Check whether the model identifies a video as an apology video from the given title and description

    Parameters:
    title (str): The title of the video
    description (str): The description of the video
    """

    question = "Does the title or description explicitly convey an apology or include language typically associated with taking responsibility, expressing regret, or asking for forgiveness?"

    return query_ollama(create_prompt(GLOBAL_CONTEXT, title, description, question))

def address_decline(title, description):
    """
    Check whether the model identifies a video as addressing a decline in channel popularity from the given title and description

    Parameters:
    title (str): The title of the video
    description (str): The description of the video
    """

    question = "Does the title or description indicate that the creator is addressing a decline in their channel popularity, such as acknowledging a decrease in views, subscribers, or engagement?"

    return query_ollama(create_prompt(GLOBAL_CONTEXT, title, description, question))

def announced_comeback(title, description):
    """
    Check whether the model identifies a video as announcing a comeback from the given title and description

    Parameters:
    title (str): The title of the video
    description (str): The description of the video
    """

    question = "Does the title or description suggest that the creator is announcing a comeback, return, or new direction for their channel, such as using phrases like 'I'm back,' 'returning,' or 'new chapter'?"

    return query_ollama(create_prompt(GLOBAL_CONTEXT, title, description, question))

def featuring_another_creator(title, description):
    """
    Check whether the model identifies a video as featuring another creator from the given title and description

    Parameters:
    title (str): The title of the video
    description (str): The description of the video
    """

    question = "Does the title or description mention or suggest that the video features another creator, such as a collaboration, guest appearance, or interview?"

    return query_ollama(create_prompt(GLOBAL_CONTEXT, title, description, question))

def clickbait(title, description):
    """
    Check whether the model identifies a video as clickbait from the given title and description

    Parameters:
    title (str): The title of the video
    description (str): The description of the video
    """

    question = "Does the title or description appear to use clickbait techniques to attract viewers, such as exagerated use of caps, misleading or exaggerated claims, sensational language, or incomplete information?"

    return query_ollama(create_prompt(GLOBAL_CONTEXT, title, description, question))

def __main__():
    # Example usage
    title = "I'm Sorry."
    description = "I made a mistake and I'm sorry. I hope you can forgive me."
    print(apology_video(title, description))

    title = "My Channel Update."
    description = "I've noticed a decrease in views and subscribers lately, so I wanted to address it in this video."
    print(address_decline(title, description))

    title = "I'm Back!"
    description = "After a long break, I'm excited to announce my return to YouTube with new content."
    print(announced_comeback(title, description))

    title = "Collab with My Friend!"
    description = "I had a great time collaborating with my friend on this video. Check it out!"
    print(featuring_another_creator(title, description))

    title = "You Won't Believe What Happened!"
    description = "This video will shock you! Watch until the end to find out."
    print(clickbait(title, description))

if __name__ == "__main__":
    __main__()