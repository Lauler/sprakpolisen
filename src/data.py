import glob
import os
import datetime as dt
import re
import pandas as pd
import logging
from nltk.tokenize import sent_tokenize
from .markdown import remove_emoji


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


def filter_dedem_comments(df):
    """
    Keep only comments with de/dem.
    """
    dedem_pattern = "(?<!\w)[Dd][Ee][Mm]?(?![\w\*])"
    df = df[df["body"].str.contains(dedem_pattern)].reset_index(drop=True)
    return df


def filter_dedem_sentences(sentences):
    """
    Keep only sentences with de/dem.
    """

    dedem_pattern = "(?<!\w)[Dd][Ee][Mm]?(?![\w\*])"
    dedem_sentences = filter(lambda sentence: bool(re.search(dedem_pattern, sentence)), sentences)

    # Don't match the string 'de och dem', as this indicatse people discussing the grammar of de vs dem.
    dedem_sentences = filter(lambda sentence: "De och dem" not in sentence, dedem_sentences)
    dedem_sentences = filter(lambda sentence: "de och dem" not in sentence, dedem_sentences)

    # Don't match single word sentences
    dedem_sentences = filter(lambda sentence: len(sentence.split()) > 1, dedem_sentences)

    return list(dedem_sentences)


def predict_dedem(comment, pipe):
    """
    Input: Sentence splitted comment.
    Keep only "de" or "dem" predictions where model has predicted
    differently from the text that was present in the comment.
    """

    de_pattern = "(?<!\w)[D][Ee](?!\w)"
    dem_pattern = "(?<!\w)[D][Ee][Mm](?!\w)"
    det_pattern = "(?<!\w)[D][Ee][Tt](?!\w)"
    enda_pattern = "(?<!\w)[Ee][Nn][Dd][Aa](?!\w)"
    anda_pattern = "(?<!\w)[Ää][Nn][Dd][Aa](?!\w)"
    comment = [re.sub(de_pattern, "de", sentence) for sentence in comment]
    comment = [re.sub(dem_pattern, "dem", sentence) for sentence in comment]
    comment = [re.sub(det_pattern, "det", sentence) for sentence in comment]
    comment = [re.sub(enda_pattern, "enda", sentence) for sentence in comment]
    comment = [re.sub(anda_pattern, "ända", sentence) for sentence in comment]
    preds = [pipe(sentence) for sentence in comment]

    pred_list = []
    for pred in preds:
        # Keep only predictions where model has predicted differently from the text that was present in the comment.
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


def date_to_epoch(year, month, day, hour=0, minute=0, second=0):
    return int(dt.datetime(year, month, day, hour, minute, second).timestamp())


def filter_de_som(sentences, preds):
    for i, data in enumerate(zip(sentences, preds)):
        sentence = data[0]
        sentence = sentence.lower()
        pred = data[1]

        for j, entity in reversed(list(enumerate(pred))):
            # expand to include " som" after "de/dem"
            expanded_entity = sentence[entity["start"] : (entity["end"] + 4)]

            if expanded_entity == "de som" or expanded_entity == "dem som":
                pred.pop(j)  # We don't want model to predict on "de/dem som"
                preds[i] = pred

    return preds


def filter_dom(sentences, preds):
    for i, data in enumerate(zip(sentences, preds)):
        pred = data[1]

        for j, entity in reversed(list(enumerate(pred))):
            if entity["word"].lower() == "dom":
                pred.pop(j)
                preds[i] = pred

    return preds


def download_submission(submission):
    logger.info(
        f"Downloading {submission.title} at {submission.permalink} with {submission.num_comments} comments."
    )
    submission.num_duplicates  # This is a hack to force the API to fetch this attribute
    submission_data = submission.__dict__

    df_sub = pd.DataFrame([submission_data])
    df_sub = df_sub[KEEP_SUBMISSION_COLUMNS]
    df_sub = df_sub.add_suffix("_sub")
    df_sub["author_sub"] = df_sub["author_sub"].apply(lambda x: x.name if x is not None else None)

    return df_sub


def download_comments(submission):
    comments = submission.comments.list()

    # Get comments
    df = pd.DataFrame([thing.__dict__ for thing in comments])
    df = df[KEEP_COMMENT_COLUMNS]
    df["id_sub"] = submission.id

    # Filter out rows with NaN in body
    df = df[df["body"].notna()].reset_index(drop=True)

    return df


def preprocess_comments(df):
    logger.info(f"Preprocessing {len(df)} comments...")

    # Remove sentences that are quotes from other comments.
    # Comments that start with ">" and end with "\n\n" are quotes.
    df["body_temp"] = df["body"].str.replace(">.*\n\n", "", regex=True)
    df["body_temp"] = df["body_temp"].str.replace("\n\n\n", "\n\n", regex=True)

    # Split comment body into list of sentences
    df["sentences"] = df["body_temp"].apply(lambda doc: sent_tokenize(doc, language="swedish"))
    df = df.drop(columns=["body_temp"])

    # Keep only sentences with de/dem
    df["sentences"] = df["sentences"].apply(lambda sen: filter_dedem_sentences(sen))
    df = df[df["sentences"].apply(lambda x: any([len(y) != 0 for y in x])) > 0].reset_index(
        drop=True
    )

    # Split sentences also on new paragraphs "\n\n" (in case someone doesn't use punctuation)
    df["sentences"] = df["sentences"].apply(lambda sens: [sen.splitlines() for sen in sens])
    # Flatten list of lists and remove empty sentences consisting of only ''.
    df["sentences"] = df["sentences"].apply(
        lambda sens: [sen for split_sens in sens for sen in split_sens]
    )
    df["sentences"] = [[sen for sen in sens if len(sen) > 0] for sens in df["sentences"]]

    # Remove emojis
    df["sentences"] = df["sentences"].apply(lambda sens: [remove_emoji(sen) for sen in sens])

    # Strip whitespace before and after sentence.
    df["sentences"] = df["sentences"].apply(lambda sens: [sen.strip() for sen in sens])

    # Remove 2 or more spaces in a row and replace by single space.
    df["sentences"] = df["sentences"].apply(
        lambda sens: [re.sub(" {2,}", " ", sen) for sen in sens]
    )

    logger.info("Finished preprocessing.")

    return df


def predict_comments(df, pipe, threshold=0.98):
    logger.info(f"Predicting with threshold {threshold}...")

    # Predict
    df["pred"] = df["sentences"].apply(lambda x: predict_dedem(x, pipe))

    # Keep only high confidence predictions
    df["pred"] = df["pred"].apply(lambda preds: filter_prediction(preds, threshold=threshold))

    # Remove "de/dem som" and "dom"
    if len(df) > 0:
        df["pred"] = df.apply(lambda x: filter_de_som(x.sentences, x.pred), axis=1)
        df["pred"] = df.apply(lambda x: filter_dom(x.sentences, x.pred), axis=1)

    return df


def count_incorrect_word(preds, word):
    """
    Only use for "det".
    """
    count = 0
    for pred in preds:
        for entity in pred:
            if len(pred) == 0:
                break

            count += 1 if (entity["word"].lower() == word and entity["entity"] != "DET") else 0

    return count


def count_all_word(preds, word):
    """
    Count all occurrences of a word in the original comment (entity["word"] is the original word).
    """
    count = 0
    for pred in preds:
        for entity in pred:
            if len(pred) == 0:
                break

            count += 1 if entity["word"].lower() == word else 0

    return count


def count_incorrect_entity(preds, word):
    count = 0
    for pred in preds:
        for entity in pred:
            if len(pred) == 0:
                break

            count += 1 if entity["entity"].lower() == word else 0

    return count


def filter_comments(df):
    logger.info("Filtering comments...")
    # Remove rows with only empty predictions (i.e. sentence preds under the threshold)
    df_comment = df[df["pred"].apply(lambda x: any([len(y) != 0 for y in x])) > 0].reset_index(
        drop=True
    )

    # Create extra variables
    df_comment["n_mis"] = df_comment["pred"].apply(lambda x: sum([len(y) for y in x]))
    df_comment["n_mis_de"] = df_comment["pred"].apply(lambda x: count_incorrect_word(x, word="de"))
    df_comment["n_mis_dem"] = df_comment["pred"].apply(
        lambda x: count_incorrect_word(x, word="dem")
    )
    df_comment["n_mis_det"] = df_comment["pred"].apply(
        lambda x: count_incorrect_entity(x, word="det")
    )
    df_comment["n_mis_enda"] = df_comment["pred"].apply(
        lambda x: count_incorrect_word(x, word="enda")
    )
    df_comment["n_mis_ända"] = df_comment["pred"].apply(
        lambda x: count_incorrect_word(x, word="ända")
    )

    df_comment["author"] = df_comment["author"].apply(lambda x: x.name if x is not None else None)
    df_comment = df_comment[df_comment["author"] != "SprakpolisenBot"]  # Filter out bot's comments
    df_comment["subreddit"] = df_comment["subreddit"].apply(
        lambda x: x.display_name if x is not None else None
    )

    if len(df_comment) > 0:
        df_comment["time_downloaded"] = int(dt.datetime.now().timestamp())

    logger.info(f"Finished filtering. {len(df_comment)} comments remaining out of {len(df)}.")

    return df_comment.reset_index(drop=True)


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

    df = df.reset_index(drop=True)
    df.to_feather(f"data/{type}/{date}_{type}.feather")


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
    df_all["edited"] = df_all["edited"].astype("int64")  # Sometimes int, sometimes boolean in API

    return df_all


def get_previous_submissions(folder="data/submission"):
    list_of_files = glob.glob(f"{folder}/*")

    # If empty folder, return empty dataframe
    if len(list_of_files) == 0:
        return pd.DataFrame()

    latest_file = max(list_of_files, key=os.path.getctime)
    df_sub = pd.read_feather(latest_file)
    return df_sub


def get_posted_comments(folder="data/posted"):
    """
    Retrieve and save posted comments to single file.
    """
    posted_files = os.listdir(folder)

    if len(posted_files) == 0:
        df_posted = pd.read_feather("data/posted_aggregated/df_posted.feather")
        df_posted.to_feather("data/df_posted.feather")
        return df_posted
    else:
        df_gen = (pd.read_feather(os.path.join(folder, file)) for file in posted_files)
        df = pd.concat(df_gen).reset_index(drop=True)

        if os.path.exists("data/posted_aggregated/df_posted.feather"):
            df_posted = pd.read_feather("data/posted_aggregated/df_posted.feather")
            df = pd.concat([df, df_posted]).reset_index(drop=True)

        df.to_feather("data/df_posted.feather")

    return df


def aggregate_posted_comments(folder="data/posted"):
    """
    Aggregate posted comments to single file.
    """
    posted_files = os.listdir(folder)
    df_gen = (pd.read_feather(os.path.join(folder, file)) for file in posted_files)
    df = pd.concat(df_gen)
    # Match pattern until _posted.feather for each file in posted_files with re.match
    posted_files = [re.match(r"(.*)_posted\.feather", file).group(1) for file in posted_files]
    posted_files = [re.sub("_", " ", file) for file in posted_files]

    posted_files = [
        re.sub(r"(\d{4}-\d{2}-\d{2}\s\d{2})-(\d{2})", r"\1:\2", file) for file in posted_files
    ]

    df["replied_time"] = posted_files
    df["replied_time"] = pd.to_datetime(df["replied_time"])

    df = df.sort_values("replied_time", ascending=False).reset_index(drop=True)

    if os.path.exists("data/posted_aggregated/df_posted.feather"):
        df_posted = pd.read_feather("data/posted_aggregated/df_posted.feather")
        df = pd.concat([df, df_posted]).reset_index(drop=True)

    dest_dir = folder + "_aggregated"
    print(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    df.to_feather(os.path.join(dest_dir, "df_posted.feather"))

    return df


def analyze_comments(submission, pipe):
    """
    Run all filters and predictions on the comments of a submission.
    """

    df_comment = download_comments(submission)
    # Regex and sentence splitting
    df_comment = preprocess_comments(df_comment)
    # Keep only comments with de/dem
    df_comment = filter_dedem_comments(df_comment)
    # Only saves preds above threshold
    df_comment = predict_comments(df_comment, pipe, threshold=0.985)
    # Keep only comments with incorrect usage of de/dem/det
    df_comment = filter_comments(df_comment)

    return df_comment
