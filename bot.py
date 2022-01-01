import os
import torch
import pandas as pd
import datetime as dt
import pprint
import logging
import praw
from psaw import PushshiftAPI
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from src.comment import choose_post, create_reply_msg
from src.data import (
    download_comments,
    download_submission,
    filter_comments,
    merge_comment_submission,
    predict_comments,
    preprocess_comments,
    save_feather,
)
from dotenv import load_dotenv

logging.basicConfig(
    filename="sprakpolisen.log",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load model
tokenizer = AutoTokenizer.from_pretrained("Lauler/deformer", model_max_length=250)
model = AutoModelForTokenClassification.from_pretrained("Lauler/deformer")
model.to(device)

# Load env variables
load_dotenv()
username = os.getenv("USR")
pw = os.getenv("PW")
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

# API
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=username,
    username=username,
    password=pw,
)

api = PushshiftAPI(reddit)

df = download_comments(api, weeks=0, hours=2, minutes=10)
df = preprocess_comments(df)  # Sentence splitting, and more
pipe = pipeline("ner", model=model, tokenizer=tokenizer, device=0)
df = predict_comments(df, pipe, threshold=0.98)  # Only saves preds above threshold
df_comment = filter_comments(df)

#### Write comment info to file ####
date = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
save_feather(df_comment, type="comment", date=date)


#### Download info about submission thread ####
df_comment["id_sub"] = df_comment["link_id"].str.slice(start=3)
df_sub = download_submission(df_comment["id_sub"].tolist(), reddit_api=reddit)
save_feather(df_sub, type="submission", date=date)

# Merge
df_all = merge_comment_submission(df_comment=df_comment, df_sub=df_sub)
df_post = choose_post(df_all, min_hour=1, max_hour=15)  # Choose which comment to post reply to
reply_msg = create_reply_msg(df_post)
save_feather(df_all, type="all", date=date)


# Reply to chosen comment
logging.info(f'Replying to comment id {df_post["id"][0]}.')
comment = reddit.comment(df_post["id"][0])
comment.reply(reply_msg)
logging.info("Succesfully replied.")

# Save replies/posted comments

df_post["replied"] = True
save_feather(df_post, type="posted", date=date)
