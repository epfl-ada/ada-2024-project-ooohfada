# Buzz, bust, and bounce-back: data-driven strategy to deal with YouTubers’ decline

<img src="https://img.asmedia.epimg.net/resizer/v2/AJQDQ4CTRRBADO5WDUJKAWFQ4A.jpg?auth=837e1df88c498b395f03b131702fa0bbb45d38d28f176a23950c3f61a588b2d9&width=360&height=203&smart=true" alt="not stonks" width="700"/>

## Data Story: https://evafrossard.github.io

## Abstract

In this project, we take on the role of **community managers** helping a YouTuber recover from a recent audience engagement crisis. May it be from a big controversy or a simple popularity decline, our goal is to **develop a data-driven strategy for navigating the aftermath of public disinterest using insights from YouTube’s ecosystem**. Drawing on patterns from previous YouTuber experiences, we’ll analyse key metrics— such as channel type, initial popularity, posting frequency—to offer tailored strategies for re-engagement. Should they change their posting strategy ? Wait a specific period before posting new content? In case of big controversies, should they issue an apology video? Our recommendations will not be based on the cause of their decline but on optimal tactics for handling its impact.

Our motivation stems from the spread of **online backlash**, impacting creators on YouTube and other platforms. We aim to provide practical insights for creators facing a **decline**, helping them make informed decisions about their next steps to rebuild audience trust or strategically embrace controversy if advantageous.

## Research Questions

- **How do we define decline for Youtubers ? How do we define a recovery ?**
- **Is a Youtube channel category affecting the effective response strategies after a decline?**
- **What timing and content strategies are best for re-engagement following public backlash? Should you change your posting strategy?**
- **In the exact case of a ‘bad buzz’, how to define it using Youtuber statistics? Are apology videos really useful?**

## Methods

We chose not to explore another dataset, considering the size of YouNiverse.
We used the following methods:
- Latent Dirichlet Allocation (LDA)
- Large Language Model (LLM)
- Pearson's Correlation
- Logistic Regression
- Propensity Score Matching
- Statistical analysis

### Before any data exploration, preprocessing of the used dataframes in the Youniverse dataset

Although the provided dataset was already cleaned, we found some heterogeneity in the dates that needed to be fixed : the sample rate is not constant and some dates are different by a few hours or a few days. To solve this:

- Use of ‘week indexes’ in df_timeseries_en.tsv and `yt_metadata_en.json`.
- Since delta_views and delta_subs are never negative even when a decrease occurs (before preprocessing, they are 0 in this case), we replaced those columns with the deltas computed from the values directly
- Augmented the time-series with the sum of likes, dislikes and views (found in yt_metadata_helper.feather) of the videos that were uploaded during the week associated with each datapoint. It is important to keep in mind that the likes, dislikes and views of the videos were all sampled in 2019 when comparing two different weeks : a recent video might have less views than an older one but still be more popular.

### Manipulation of the dataset, and discovery analysis

Initially, we had in mind to explore and mathematically detect **‘bad buzz’** or controversy only.

To do so, we followed the below process:

- Creation of a dataset containing **only known Youtube channels that underwent a bad buzz**: `df_bb_timeseries_en.tsv`.
- Qualify what a ‘bad buzz’ is from data, and see if we could map the date of their bad buzz to specific data points.
  - **First attempt**: look for losses of subs, as we assumed that it was the general consequence of a bad buzz. We were able to conclude that a Youtube channel will rarely ‘lose’ subscribers due to the increasing traffic on the platform.
  - **Second attempt**: we used reduction or stagnation of growth coefficient of subscribers. We were able to graphically compare the delta subs to the rolling growth average of the growth coefficient and determine the moments where both diverged.

However, the bad buzz dataset containing only around 40 Youtubers, we were afraid that it would not allow us to pursue ML techniques (regression, mapping..). Therefore, we made the decision to open up our subject and tackle **“viewership decline"** instead of only big controversy.

The following analysis was performed:

- General and **visual overview** of the whole dataset through the prism of viewership declines
- For each decline in growth, observe around those timestamps what was happening concerning like/ratios, views, activities (number of posted videos) to see if any correlation could be found (using p-values...), which could lead to a pertinent ML analysis.

## Analysis of the declines
### Definition of a decline and a recovery
#### Decline
Since very few Youtube channels actually face a loss of subscribers, we decided to look for **slowdowns in the growth rate** of subscribers in order to detect declines. We used the rolling growth average as a baseline to look for periods of time where the actual growth is below the rollig averagge for more than 8 consecutive weeks and/or there is a growth difference greater than 80%.
#### Recovery
We decided to consider a recovery as successful if it happens within 16 weeks after the start of the decline

### Factors to analyze
- Channel categories
- Number of views and subscribers
- Upload frequecy
- Video Duration

### Reactions to a decline
In this part of the project, we used **Propensity Score Matching** to ensure fair comparisons between the different strategies and to see which one yields the best recovery rate. We also performed **t-tests** to see if the observed differences in recovery rates were statistically significant.

#### Posting frequency
We tried to see if it is better for a youtuber to post less, keep uploading the same content or be more active after a decline. 

#### Video Duration
Similarly as for the posting frequency, we made a distinction between three possible strategies: posting shorter videos, longer videos, or videos of the same length.

#### Topic Change
We also investigated to see if a topic transition could be beneficial for Youtubers facing a decline. In order to identify the thematics of the youtube channels before and after their declines, we used the **Latent Dirichlet Allocation** to distinguish 20 topics of 15 words among all the tags of the videos and we then used a **Large Language Model** to assign a title to each topic. With this categorization, we were able to identify the channels that changed the topic of their videos after a decline and we analyzed the recovery rate of these videos.

## Big Youtubers in a Crisis
In this part of the project, the analysis is focused on the Youtubers with more than **1 million subscribers**. We started by comparing their recovery rates with the rest of the youtubers, and we then proceeded with an analysis of the video titles with the **Large Language Model** **Mistral** from the **OLLAMA** project in order to see if each video belongs to the following categories:
- Apology videos
- Videos addressing the decline
- Comeback announcements
- Break announcements
- Collaboration videos
- Clickbait videos
We then compared the effects of these different strategies by leveraging **Propensity Score Matching** one more time.

In the aim of using an **on-device LLM** to analyse the metadata of videos that follow a popularity crisis, we applied a special treatment to the large `yt_metadata_en.json` dataset, since it contains the title and description of all crawled videos. To make it usable, we:

- split it into smaller, handable chunks
- got rid of the unneeded columns
- preprocessed them by indexing by [“channel”, “week”]
- deleted the rows that contained missing values.
- kept track of what channel appeared in each chunk in `channel_chunk_dict.json` to make them more accessible.

## Contributions within the team

- **Martina**: Worked on the initial analysis of declines, identification of the decline events, the analysis of the recovery rates of the Youtubers that changed their posting frequency and mean video duration, and the repository structure.

- **Nathan**: Worked on the Propensity Score Matching analysis, the recoveery analysis, the analysis of the recovery rates of the Youtubers that changed their posting frequency and mean video duration, and the repository structure.
- **Eva**: Worked on initial analysis of declines, NLP for topic identification, topic changes analysis.
- **Nicolas**: Worked on the identification of the decline events, the LLM for the analysis of the video titles, the analysis of the proportions and recovery rates of the Youtubers with more than 1 million subscribers depending on strategy applied, and the repository structure.
- **Pauline**: Worked on the recovery analysis, the entire structure + visual + redaction of the data story.

## Repository structure

```
ADA-2024-PROJECT-OOOHFADA/
├── data/                                           # Data folder (will contain all the data used and created in the project)
├── extras/                                         # Extra files
│   ├── dataset_description.md                      # Description of the dataset
│   └── dataset_presenation.pdf                     # Presentation of the dataset            
├── plot_data/                                      # Folder for plots (will contain the plots generated by the scripts 
│                                                   # and used in the data story when generated by the results notebook)
├── src/                            
│   ├── data/                                       # Data processing scripts
│   │   ├── bbdataset_preprocessing.py              # Preprocessing of the bad buzz dataset
│   │   ├── dataloader_functions.py                 # Dataloader for the dataset
│   │   ├── final_dataset_creation.py               # Creation of the final dataset
│   │   ├── preprocessing.py                        # Preprocessing of the dataset
│   │   ├── reduce_metadata.py                      # Reduction of the metadata dataset
│   │   └── video_extraction.py                     # Extraction of the videos
│   ├── models/                                     # LLM related scripts
│   │   └── llm_call_helpers.py                     # LLM call helpers
│   ├── scripts/                                    # Utility scripts
│   │   └──  preprocessing_pipeline.sh              # Preprocessing pipeline script
│   ├── utils/                                      # Helper functions
│   │   ├── 1M_plus_utils.py                        # Helper functions for the 1M+ analysis
│   │   ├── find_video_categories.py                # Helper functions for the video categories analysis
│   │   ├── plots.py                                # Helper functions for the plots
│   │   ├── recovery_analysis_utils.py              # Helper functions for the recovery analysis
│   │   └── results_utils.py                        # Helper functions for the results
├── venv/                                           # Virtual environment
├── .gitignore
├── pip_requirements.txt                            # Required packages
├── README.md                                       # Project description and instructions
└── results.ipynb                                   # Jupyter notebook with the results
```

## How to execute the code

**1.** Clone the repository:
```bash
git clone https://github.com/epfl-ada/ada-2024-project-ooohfada.git 
cd ada-2024-project-ooohfada
```
**2.** Download and Install [OLLAMA](https://ollama.com)  
:warning: Make sure the app is running in background before creating the virtual environment
**3.** Create a virtual environment and install the required packages using the `pip_requirements.txt` file
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r pip_requirements.txt
```
**4.** Set up the LLM by pulling [MISTRAL model](https://ollama.com/library/mistral) from the OLLAMA API
```bash
ollama pull mistral
```
**5.** Download the dataset from the [ADA 2024 YouNiverse Kaggle page](https://zenodo-org.translate.goog/records/4650046?_x_tr_sl=en&_x_tr_tl=fr&_x_tr_hl=fr&_x_tr_pto=sc) and place it in the `data/` folder  

**6.** Run the `results.ipynb` notebook to generate all the results and plots  
   :warning: The notebook is quite long and some cells take a while (many hours) to run (precised above the cells)
   :warning: The notebook is also memory intensive, make sure you have enough RAM to run it (32GB recommended)

