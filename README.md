 # Buzz, bust, and bounce-back: data-driven strategy to deal with YouTubers’ decline

## Abstract

In this project, we take on the role of community managers helping a YouTuber recover from a recent popularity crisis. May it be from a big controversy or a simple popularity struggle, our goal is to develop a data-driven strategy for navigating the aftermath of public disinterest using insights from YouTube’s ecosystem. Drawing on patterns from previous YouTuber experiences, we’ll analyse key metrics— such as channel type, initial popularity, creator’s perceived integrity—to offer tailored strategies for re-engagement. Should they change their posting strategy ? Wait a specific period before posting new content? In case of big controversies, should they issue an apology video? Our recommendations will not be based on the cause of their decline but on the optimal tactics for handling its impact.

Our motivation stems from the rapid spread of online backlash, which impacts creators on YouTube and other platforms. We aim to provide practical insights for creators facing a decline, helping them make informed decisions about their next steps to rebuild audience trust or strategically embrace controversy if advantageous.

## Research Questions

- How do we define decline for Youtubers ? 
- Is a Youtube channel category affecting the effective response strategies after a decline?
- What timing and content strategies are best for re-engagement following public backlash?
- In the exact case of a ‘bad buzz’, how to define it using Youtuber statistics? Are apology videos really useful ? 

## Methods

The main objective of our P2 was to explore our dataset and programmatically detect what a “viewership decline" is. To do so, we used the following methods: 

### Before any data exploration, preprocessing of the used dataframes in the Youniverse dataset

Although the provided dataset was already cleaned, we found that there was some heterogeneity in the dates that needed to be fixed : the sample rate is not constant and some dates are different by a few hours or a few days. To solve this: 
- Use of ‘week indexes’ in df_timeseries_en.tsv and `yt_metadata_en.json`. 
- Since delta_views and delta_subs are never negative even when a decrease occurs (before preprocessing, they are 0 in this case), we replaced those columns with the deltas computed from the values directly
- Augmented the time-series with the sum of likes, dislikes and views (found in yt_metadata_helper.feather) of the videos that were uploaded during the week associated with each datapoint. It is important to keep in mind that the likes, dislikes and views of the videos were all sampled in 2019 when comparing two different weeks : a recent video might have less views than an older one but still be more popular. 

### Manipulation of the dataset, and discovery analysis

Initially, we had in mind to explore and mathematically detect ‘bad buzz’ or controversy only. 

To do so, we followed the below process:

- Creation of a dataset containing only known Youtube channels that underwent a bad buzz: `df_bb_timeseries_en.tsv`.
- Qualify what a ‘bad buzz’ is from data, and see if we could map the date of their bad buzz to specific data points.
    - First attempt: look for losses of subs, as we assumed that it was the general consequence of a bad buzz. We were able to conclude that a Youtube channel will rarely ‘lose’ subscribers due to the increasing traffic on the platform.
    - Second attempt: we used reduction or stagnation of growth coefficient of subscribers. We were able to graphically compare the delta subs to the rolling growth average of the growth coefficient and determine the moments where both diverged. 

However, the bad buzz dataset containing only around 40 Youtubers, we were afraid that it would not allow us to pursue ML techniques like regression, … and actually have coherent data to base our analysis on. Therefore, we made the decision to open up our subject and tackle “viewership decline’’ instead of only big controversy. This particular topic will be tackled in a separate alternative discussion in our P3. After creating a new dataset with only the data name of new dataset from channels that underwent a popularity struggle, we performed numerous exploration processes on this newly made dataset, containing now only Youtube channels that underwent a persistent decline (state number of weeks acceptable for threshold) using the following techniques: 
- method 1
- method 2



### LLMs
To enrich our analysis and in the aim of using an on-device LLM to analyse the metadata of videos that follow a popularity crisis and detect ###TODO NICOLAS###, we started the processing of video metadata. To make the file usable, we split it into smaller chunks, got rid of the columns that we do not plan on using and then preprocessed them by indexing by [“week”, “channel”], deleting the rows that contained NaNs. In the file ###TODO NICO###, we kept track of what channel appeared in each chunk to make their use easier.
LLMs will be used to detect potential apology videos in the alternative discussion about bad buzz. On a broader aspect, it will also be used ……


## Proposed timeline and organisation within the team

### Week 1 (18/11 - 24/11)
1. Finalise method to find channels of interest (Eva + Martina + Pauline)
2. Identify/create the metrics that characterise the channel’s popularity (magnitude, time scale) => used for recovery level after a fall down. (Eva + Martina + Pauline)
3. Identify the behaviour/reaction from the channel metrics to differentiate channels which failed to recover from those that managed to recover. (Nathan + Nicolas)
### Week 2 (25/11 - 01/12)
4. Matched observational study using propensity score to find out which factors have the greatest impact on recovery. (All since it is a crucial part)
### Week 3 (02/12 - 08/12)
5. Perform a regression of the recovery level on the channel’s reaction (All since it is a crucial part)
### Week 4/5 (09/12 - 19/12)
6. Data story and repository finalization (All but we don't know the repartition)

