import logging
import os
import torch
import praw
import pandas as pd
import datetime as dt
from dotenv import load_dotenv
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)
from src.comment import choose_post, create_reply_msg
from src.data import (
    analyze_comments,
    download_submission,
    get_posted_comments,
    merge_comment_submission,
    save_feather,
)
from src.translate import translation_preprocess

logging.basicConfig(
    filename="sprakpolisen.log",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#### Load models
tokenizer = AutoTokenizer.from_pretrained("Lauler/deformer", model_max_length=250)
model = AutoModelForTokenClassification.from_pretrained("Lauler/deformer")
model.to(device)

# NER pipeline
pipe = pipeline("ner", model=model, tokenizer=tokenizer, device=0)

# Machine Translation model
tokenizer_translate = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-sv-en")
model_translate = AutoModelForSeq2SeqLM.from_pretrained(
    "Helsinki-NLP/opus-mt-sv-en", output_attentions=True
)
model_translate.eval()
model_translate.to(device)

#### Load env variables
load_dotenv()
username = os.getenv("USR")
pw = os.getenv("PW")
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

#### API
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=username,
    username=username,
    password=pw,
)

subreddit = reddit.subreddit("sweden")


df_subs = []
df_comments = []

for submission in subreddit.hot(limit=35):
    if submission.num_comments == 0:
        continue

    df_sub = download_submission(submission)
    df_comment = analyze_comments(submission, pipe=pipe)
    df_subs.append(df_sub)
    df_comments.append(df_comment)

df_sub = pd.concat(df_subs).reset_index(drop=True)
df_comment = pd.concat(df_comments).reset_index(drop=True)

#### Write comment and submission info to file ####
date = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
save_feather(df_comment, type="comment", date=date)
save_feather(df_sub, type="submission", date=date)

# Merge
df_all = merge_comment_submission(df_comment=df_comment, df_sub=df_sub)

try:
    df_history = get_posted_comments()  # Get SprakpolisenBot's previous replies to comments
    # Don't post twice in same thread
    df_all = df_all[~df_all["link_id"].isin(df_history["link_id"])].reset_index(drop=True)
except:
    pass

df_all = df_all[~(df_all["n_mis_det"] == 1)].reset_index(drop=True)

# Choose which comment to post reply to
df_post = choose_post(df_all, min_hour=0.7, max_hour=19)

df_all.columns
# df_post = df_all.iloc[1:2].reset_index(drop=True)

df_post["sentences"] = df_post["sentences"].apply(
    lambda sens: [sen.replace("…", ".") for sen in sens]
)

#### Translate to English
pipes = translation_preprocess(
    df_post,
    model_translate=model_translate,
    tokenizer_translate=tokenizer_translate,
    device=device,
)

reply_msg = create_reply_msg(df_post, pipes=pipes)


save_feather(df_all, type="all", date=date)

for i in range(len(df_all)):
    try:
        # Reply to chosen comment
        logging.info(f'Replying to comment id {df_post["id"][0]}.')
        comment = reddit.comment(df_post["id"][0])
        comment.reply(body=reply_msg)
        break
    except Exception as e:
        if isinstance(e, praw.exceptions.RedditAPIException):
            # Due to incredibly stupid changes on reddit around how blocked comments work,
            # SprakpolisenBot may be blocked from replying to anyone in a comment chain
            # if a single comment author in the comment chain has blocked SprakpolisenBot.
            logging.error(f'Failed replying to comment id {df_post["id"][0]} because of block.')
            df_all = df_all[df_all["id"] != df_post["id"][0]]  # Remove unsuccessful reply attempt
            df_post = choose_post(df_all, min_hour=1, max_hour=19)

            #### Translate to English
            pipes = translation_preprocess(
                df_post,
                model_translate=model_translate,
                tokenizer_translate=tokenizer_translate,
                device=device,
            )
            reply_msg = create_reply_msg(df_post, pipes=pipes)


logging.info("Succesfully replied.")

# Save replies/posted comments
df_post["replied"] = True
df_post["replied_time"] = pd.to_datetime(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
save_feather(df_post, type="posted", date=date)
