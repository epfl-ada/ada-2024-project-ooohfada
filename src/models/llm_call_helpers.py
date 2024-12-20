import ollama
from tqdm import tqdm

GLOBAL_CONTEXT = "You are analyzing YouTube videos to determine how creators handle a decline in their channel popularity. \n The video is analyzed based on its title. Your task is to answer the given specific question."
TOPIC_CONTEXT = "You are analyzing YouTube channels to determine the topic based on the distribution of the 15 main tokens found with Latent Dirichlet Allocation (LDA). Your task is to identify the name of the topic based on the given distribution."

def llm_pull(model):
    """
    Pulls the model from the Ollama API

    Parameters:
    model (str): The name of the model to pull
    """
    ollama.pull(model)

def query_ollama(prompt, model="mistral:latest"):
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
    try:
        response = ollama.generate(model=model, prompt=prompt)
        raw_response = response.get('response')
        clean_response = raw_response.split()[0].replace(".", "").replace(",", "") # Keep only the first word and delete any ponctuation
        return clean_response
    except Exception as e:
        raise RuntimeError(f"Error querying the local model: {e}")

def create_prompt(context, title, question):
    """
    Construct a prompt with context, video details, and a specific question.

    Parameters:
    context (str): The context of the question
    title (str): The title of the video
    question (str): The question to ask

    Returns:
    str: The constructed prompt
    """
    return (
        f"Context:\n"
        f"{context}\n\n"
        f"Video Title:\n"
        f"{title}\n\n"
        f"Question:\n"
        f"{question}\n\n"
        f"Answer with 'True' or 'False' only."
    )

def create_prompt_topic(context, main_tokens_distribution):
    """
    Construct a prompt with context and the distribution of the 15 main tokens found with Latent Dirichlet Allocation (LDA)

    Parameters:
    context (str): The context of the question
    main_tokens_distribution: The distribution of the 15 main tokens found with LDA 

    Returns:
    str: The constructed prompt
    """
    return (
        f"Context:\n"
        f"{context}\n\n"
        f"Main Tokens Distribution:\n"
        f"{main_tokens_distribution}\n\n"
        f"Output only the name of the topic without any sentences (1-2 words)."
    )

def apology_video(title):
    """
    Check whether the model identifies a video as an apology video from the given title

    Parameters:
    title (str): The title of the video
    """

    question = "Does the title explicitly convey an apology or include language typically associated with taking responsibility, expressing regret, or asking for forgiveness?"

    return query_ollama(create_prompt(GLOBAL_CONTEXT, title, question))

def address_decline(title):
    """
    Check whether the model identifies a video as addressing a decline in channel popularity from the given title

    Parameters:
    title (str): The title of the video
    """

    question = "Does the title indicate that the creator is addressing a decline in their channel popularity, such as acknowledging a decrease in views, subscribers, or engagement?"

    return query_ollama(create_prompt(GLOBAL_CONTEXT, title, question))

def announced_comeback(title):
    """
    Check whether the model identifies a video as announcing a comeback from the given title

    Parameters:
    title (str): The title of the video
    """

    question = "Does the title suggest that the creator is announcing a comeback, return, or new direction for their channel, such as using phrases like 'I'm back,' 'returning,' or 'new chapter'?"

    return query_ollama(create_prompt(GLOBAL_CONTEXT, title, question))

def announced_break(title):
    """
    Check whether the model identifies a video as announcing a break from the given title

    Parameters:
    title (str): The title of the video
    """

    question = "Does the title suggest that the creator is announcing a break, hiatus, or temporary pause in content creation, such as using phrases like 'taking a break,' 'on hiatus,' or 'need time off'?"

    return query_ollama(create_prompt(GLOBAL_CONTEXT, title, question))

def featuring_another_creator(title):
    """
    Check whether the model identifies a video as featuring another creator from the given title

    Parameters:
    title (str): The title of the video
    """

    question = "Does the title mention or suggest that the video features another creator, such as a collaboration, guest appearance, or interview?"

    return query_ollama(create_prompt(GLOBAL_CONTEXT, title, question))

def clickbait(title):
    """
    Check whether the model identifies a video as clickbait from the given title

    Parameters:
    title (str): The title of the video
    """

    question = "Does the title appear to use clickbait techniques to attract viewers, such as exagerated use of caps, misleading or exaggerated claims, sensational language, or incomplete information?"

    return query_ollama(create_prompt(GLOBAL_CONTEXT, title, question))

def apply_llm_detection(videos, verbose = True):
    """
    Apply the LLM detection functions to a DataFrame of video titles
    
    Parameters:
    videos (pd.DataFrame): A DataFrame containing video titles
    
    Returns:
    pd.DataFrame: A DataFrame with the detection results appended
    """

    if verbose:
        print("Applying LLM detection functions ...")

    tqdm.pandas()

    videos['apology'] = videos['title'].progress_apply(lambda x: apology_video(x))
    videos['decline_addressed'] = videos['title'].progress_apply(lambda x: address_decline(x))
    videos['comeback'] = videos['title'].progress_apply(lambda x: announced_comeback(x))
    videos['break'] = videos['title'].progress_apply(lambda x: announced_break(x))
    videos['featuring'] = videos['title'].progress_apply(lambda x: featuring_another_creator(x))
    videos['clickbait'] = videos['title'].progress_apply(lambda x: clickbait(x))

    if verbose:
        print("LLM detection functions applied.\n")

    if verbose:
        print("Saving the results ...")

    videos.to_csv('data/llm_strategies.csv')

    if verbose:
        print("Results saved to 'data/llm_strategies.csv'.\n")

def clean_strategies_results(results):
    if results == True or results == 'True':
        return int(1)
    elif results == False or results == 'False':
        return int(0)
    else:
        return None

def generate_topic_name(main_tokens_distribution):
    """
    Identify the topic name based on the distribution of the 15 main tokens found with Latent Dirichlet Allocation (LDA)

    Parameters:
    main_tokens_distribution: The distribution of the 15 main tokens found with LDA 

    Returns:
    str: The topic name generated by the LLM
    """

    return query_ollama(create_prompt_topic(TOPIC_CONTEXT, main_tokens_distribution))

def __main__():

    # Test the functions
    title = "I'm Sorry."
    print(apology_video(title))

    title = "My Channel Update."
    print(address_decline(title))

    title = "I'm Back!"
    print(announced_comeback(title))

    title = "Collab with My Friend!"
    print(featuring_another_creator(title))

    title = "You Won't Believe What Happened!"
    print(clickbait(title))

if __name__ == "__main__":
    __main__()