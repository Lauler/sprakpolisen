import logging
import datetime as dt
import pandas as pd
import time

"""
Legacy code from the first version of the bot based on Pushshift.
"""

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


def start_time(weeks=0, days=0, hours=2, minutes=10):
    """
    How many days, hours and minutes back to query reddit for comments.
    """
    start_time = dt.datetime.now() - dt.timedelta(
        weeks=weeks, days=days, hours=hours, minutes=minutes
    )
    return int(start_time.timestamp())


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


def download_comments_between(api, start_time, end_time, q="de|dem", subreddit="sweden"):
    logger.info(f"Downloading all comments from /r/swdeden from {start_time} to {end_time}.")

    gen = api.search_comments(after=start_time, before=end_time, q=q, subreddit=subreddit)

    # Get comments
    df = pd.DataFrame([thing.__dict__ for thing in gen])
    df = df[KEEP_COMMENT_COLUMNS]

    return df


def download_submission(link_ids, reddit_api, backoff_factor=0.4):
    link_ids = list(set(link_ids))
    df_list = []
    for link_id in link_ids:
        for i in range(5):
            # Exponential backoff
            backoff_time = backoff_factor * (2**i)

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
