import os
import time
import datetime as dt
import re
import pandas as pd
import logging
from nltk.tokenize import sent_tokenize
from .utils import join_insertion


logger = logging.getLogger(__name__)

KEEP_COMMENT_COLUMNS = [
    "id",
    "link_id",
    "score",
    "author",
    "body",
    "subreddit_id",
    "permalink",
    "edited",
    "ups",
    "num_reports",
    "total_awards_received",
    "subreddit",
    "gilded",
    "can_mod_post",
    "send_replies",
    "parent_id",
    "author_fullname",
    "downs",
    "collapsed",
    "is_submitter",
    "body_html",
    "collapsed_reason",
    "collapsed_reason_code",
    "stickied",
    "unrepliable_reason",
    "score_hidden",
    "locked",
    "name",
    "created",
    "created_utc",
    "subreddit_name_prefixed",
    "controversiality",
    "collapsed_because_crowd_control",
    "mod_note",
    "_fetched",
]

KEEP_SUBMISSION_COLUMNS = [
    "author",
    "author_flair_text",
    "created",
    "created_utc",
    "gilded",
    "id",
    "is_meta",
    "is_self",
    "is_video",
    "link_flair_text",
    "locked",
    "mod_note",
    "name",
    "num_comments",
    "num_crossposts",
    "num_duplicates",
    "permalink",
    "pinned",
    "removal_reason",
    "score",
    "selftext",
    "selftext_html",
    "stickied",
    "title",
    "ups",
    "upvote_ratio",
    "url",
]


def filter_dedem_sentences(sentences):
    """
    Keep only sentences with de/dem.
    """

    dedem_pattern = "(?<!\w)[Dd][Ee][Mm]?(?!\w)"
    dedem_sentences = filter(lambda sentence: bool(re.search(dedem_pattern, sentence)), sentences)

    return list(dedem_sentences)


def predict_dedem(comment, pipe):
    """
    Input: Sentence splitted comment. 
    Keep only "de" or "dem" predictions where model has predicted
    differently from the text that was present in the comment.
    """

    de_pattern = "(?<!\w)[D][Ee](?!\w)"
    dem_pattern = "(?<!\w)[D][Ee][Mm](?!\w)"
    comment = [re.sub(de_pattern, "de", sentence) for sentence in comment]
    comment = [re.sub(dem_pattern, "dem", sentence) for sentence in comment]
    preds = [pipe(sentence) for sentence in comment]

    pred_list = []
    for pred in preds:
        pred_list.append(
            [d for d in pred if (d["entity"] != "ord") and (d["entity"].lower() != d["word"])]
        )

    return pred_list


def filter_prediction(preds, threshold=0.9):
    """
    Keep only predictions above confidence threshold.
    """
    keep_preds = []
    for sentence_preds in preds:
        if len(sentence_preds) > 0:
            keep_preds.append([pred for pred in sentence_preds if pred["score"] > threshold])
        else:
            keep_preds.append([])

    return keep_preds


def start_time(weeks=0, days=0, hours=2, minutes=10):
    """
    How many days, hours and minutes back to query reddit for comments.
    """
    start_time = dt.datetime.now() - dt.timedelta(
        weeks=weeks, days=days, hours=hours, minutes=minutes
    )
    return int(start_time.timestamp())


def filter_de_som(sentences, preds):
    for i, data in enumerate(zip(sentences, preds)):
        sentence = data[0]
        sentence = sentence.lower()
        pred = data[1]

        for entity in pred:
            # expand to include " som" after "de/dem"
            expanded_entity = sentence[entity["start"] : (entity["end"] + 4)]

            if expanded_entity == "de som" or expanded_entity == "dem som":
                preds[i] = []  # We don't want model to predict on "de/dem som"

    return preds


def download_comments(api, weeks=0, days=0, hours=2, minutes=10):
    logger.info(
        f"Downloading all comments from /r/swdeden from the last {weeks} weeks, {days} days, {hours} hours and {minutes} minutes."
    )
    # Download comments this far back in time.
    after_time = start_time(weeks=weeks, days=days, hours=hours, minutes=minutes)
    gen = api.search_comments(after=after_time, q="de|dem", subreddit="sweden")

    # Get comments
    df = pd.DataFrame([thing.__dict__ for thing in gen])
    df = df[KEEP_COMMENT_COLUMNS]

    return df


def preprocess_comments(df):
    logger.info(f"Preprocessing {len(df)} comments...")
    # Split comment body into list of sentences
    df["sentences"] = df["body"].apply(lambda doc: sent_tokenize(doc, language="swedish"))

    # Keep only sentences with de/dem
    df["sentences"] = df["sentences"].apply(lambda sen: filter_dedem_sentences(sen))
    df = df[df["sentences"].apply(lambda x: any([len(y) != 0 for y in x])) > 0].reset_index(
        drop=True
    )

    # Split sentences also on new paragraphs "\n\n"
    df["sentences"] = df["sentences"].apply(lambda sens: [sen.splitlines() for sen in sens])
    # Flatten list of lists and remove empty sentences consisting of only ''.
    df["sentences"] = df["sentences"].apply(
        lambda sens: [sen for split_sens in sens for sen in split_sens]
    )
    df["sentences"] = [[sen for sen in sens if len(sen) > 0] for sens in df["sentences"]]

    # Remove sentences that are quotes from other comments (comments that start with ">")
    df["sentences"] = df["sentences"].apply(
        lambda sens: [sen for sen in sens if not sen[0] == ">"]
    )

    logger.info("Finished preprocessing.")

    return df


def predict_comments(df, pipe, threshold=0.98):
    logger.info(f"Predicting with threshold {threshold}...")

    # Predict
    df["pred"] = df["sentences"].apply(lambda x: predict_dedem(x, pipe))

    # Keep only high confidence predictions
    df["pred"] = df["pred"].apply(lambda preds: filter_prediction(preds, threshold=threshold))

    # Remove de/dem som
    if len(df) > 0:
        df["pred"] = df.apply(lambda x: filter_de_som(x.sentences, x.pred), axis=1)

    return df


def count_incorrect(preds, word):
    count = 0
    for pred in preds:
        for entity in pred:
            if len(pred) == 0:
                break

            count += 1 if entity["word"] == word else 0

    return count


def filter_comments(df):
    logger.info("Filtering comments...")
    # Remove rows with only empty predictions (i.e. sentence preds under the threshold)
    df_comment = df[df["pred"].apply(lambda x: any([len(y) != 0 for y in x])) > 0].reset_index(
        drop=True
    )

    # Create extra variables
    df_comment["nr_mistakes"] = df_comment["pred"].apply(lambda x: sum([len(y) for y in x]))
    df_comment["nr_mistakes_de"] = df_comment["pred"].apply(
        lambda x: count_incorrect(x, word="de")
    )
    df_comment["nr_mistakes_dem"] = df_comment["pred"].apply(
        lambda x: count_incorrect(x, word="dem")
    )
    df_comment["author"] = df_comment["author"].apply(lambda x: x.name if x is not None else None)
    df_comment = df_comment[df_comment["author"] != "SpråkpolisenBot"]  # Filter out bot's comments
    df_comment["subreddit"] = df_comment["subreddit"].apply(
        lambda x: x.display_name if x is not None else None
    )

    if len(df_comment) > 0:
        df_comment["time_downloaded"] = int(dt.datetime.now().timestamp())

    logger.info(f"Finished filtering. {len(df_comment)} comments remaining out of {len(df)}.")

    return df_comment


def save_feather(df, type, date):
    os.makedirs(f"data/{type}", exist_ok=True)

    try:
        assert len(df) > 0
    except AssertionError as e:
        logger.exception("Dataframe is empty. No incorrect usage of de/dem found.")
        raise e

    logger.info(f"Saving comments to data/{type}/{date}_{type}.feather")

    if type == "comment" or type == "all":
        df["edited"] = df["edited"].astype("int64")

    df.to_feather(f"data/{type}/{date}_{type}.feather")


def download_submission(link_ids, reddit_api, backoff_factor=0.4):

    link_ids = list(set(link_ids))
    df_list = []
    for link_id in link_ids:
        for i in range(5):
            # Exponential backoff
            backoff_time = backoff_factor * (2 ** i)

            try:
                submission = reddit_api.submission(id=link_id)
                logger.info(f"Downloading {submission.title} at {submission.permalink}")
                submission_data = vars(submission)
                submission_cols = {key: submission_data[key] for key in KEEP_SUBMISSION_COLUMNS}
                df_list.append(submission_cols)
                break

            except:
                logger.error(f"Download of {link_id} failed. Retry {i}.")

            time.sleep(backoff_time)

    df_sub = pd.DataFrame(df_list)
    df_sub = df_sub.add_suffix("_sub")
    df_sub["author_sub"] = df_sub["author_sub"].apply(lambda x: x.name if x is not None else None)

    return df_sub


def merge_comment_submission(df_comment, df_sub):
    df_all = df_comment.merge(
        df_sub[
            [
                "id_sub",
                "created_sub",
                "link_flair_text_sub",
                "locked_sub",
                "num_comments_sub",
                "permalink_sub",
                "title_sub",
                "ups_sub",
                "upvote_ratio_sub",
            ]
        ],
        how="left",
        on="id_sub",
    )

    df_all["hours_after_thread_post"] = (df_all["created"] - df_all["created_sub"]) / 3600
    df_all["hours_since_post"] = (dt.datetime.now().timestamp() - df_all["created"]) / 3600
    df_all["hours_age_thread"] = (dt.datetime.now().timestamp() - df_all["created_sub"]) / 3600
    df_all["replied"] = False  # We have not yet replied to any of the posts

    return df_all
