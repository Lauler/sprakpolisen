import logging

from .markdown import (
    add_horizontal_rule,
    add_paragraph,
    add_quotation,
    create_analysis_legend,
    create_guide_en,
    create_footer,
    create_guide,
    create_header,
    insert_heading,
    wrongful_de_dem,
)


logger = logging.getLogger(__name__)


def choose_post(df_all, min_hour, max_hour):
    """
    Choose which comment to post reply.
    """

    logger.info(
        "Choosing comment to reply to: Filtering out 'seriös' threads and too young/old threads."
    )
    # Filter away any threads with "seriös" tag or "seriös" in submission title.
    # Filter away comments in threads younger than 1h and older than 15h.
    df_all = df_all[~(df_all["link_flair_text_sub"].str.lower() == "seriös")]
    df_all = df_all[~(df_all["title_sub"].str.lower().str.contains("seriös"))]
    df_all = df_all[
        (df_all["hours_age_thread"] > min_hour) & (df_all["hours_age_thread"] < max_hour)
    ]
    df_all = df_all[~df_all["locked_sub"]]  # Thread should not be locked
    df_all = df_all.sort_values("hours_age_thread").reset_index(drop=True)

    try:
        assert len(df_all) > 0
    except AssertionError as e:
        logger.exception("No suitable comment reply candidates. Exiting.")
        raise e

    if any(df_all["nr_mistakes"] > 1):
        df_multimistake = df_all[df_all["nr_mistakes"] > 1].reset_index(drop=True)

        if len(df_multimistake) > 1:
            max_mistake_idx = df_multimistake["nr_mistakes"].idxmax()
            df_post = df_multimistake.iloc[max_mistake_idx : (max_mistake_idx + 1), :]
        else:
            df_post = df_multimistake
    else:
        df_post = df_all.iloc[0:1, :]

    df_post = df_post.reset_index(drop=True)
    return df_post


def match_case(sentence, entity):
    """
    Match lower/upper case for incorrect/corrected version of word.
    """

    word = sentence[entity["start"] : entity["end"]]
    correct_word = entity["entity"].lower()

    if word.islower():
        pass  # correct_word was made lower cased before
    elif word.isupper():
        correct_word = correct_word.upper()
    elif word[0].isupper():
        word_list = list(correct_word)
        word_list[0] = correct_word[0].upper()
        correct_word = "".join(word_list)

    # We ignore handling "dE, dEM, deM, dEM", and in such cases just return lower cased version
    return word, correct_word


def correct_sentence(preds, sentences):
    offset = 0
    correct_sens = []
    for pred, sentence in zip(preds, sentences):

        if len(pred) == 0:
            continue

        original_sentence = sentence

        for entity in pred:
            original_word, correct_word = match_case(original_sentence, entity)
            begin_idx = entity["start"] + offset
            end_idx = entity["end"] + offset
            score = f'{entity["score"]:.2%}'  # score as percentage with 2 decimals
            sentence = (
                f"{sentence[:begin_idx]}~~{original_word}~~ **{correct_word}**{sentence[end_idx:]}"
            )

            # Count all added text: words, strikethroughs (~~), spaces, parenthesis and bold (**).
            # Keep track of the offset in case of multiple de/dem errors in same sentence.
            offset += 9 + len(correct_word)  # + len(score)

        correct_sens.append(sentence)
        offset = 0

    return correct_sens


def correct_sentence_en(preds, correct_sens):
    """
    We have correctly translated sentences already, but want to 
    introduce the wrong form of they/them with a strikethrough
    next to the already corrected instance of they/them/the/those.
    """
    corrected_sv = preds[0]
    preds_en = preds[1]
    offset = 0
    subtract_pad = -5  # <pad> token length needs to be subtracted from decoder string indices
    mistake_sens = []

    they_words = [
        "They",
        "they",
        "Those",
        "those",
        "The",
        "the",
        "These",
        "these",
    ]  # Can be substituted for 'de'
    them_words = ["Them", "them"]

    for i, (pred, sentence) in enumerate(zip(preds_en, correct_sens)):

        target_indices = []  # To check if we're trying to correct same word twice or more
        contains_pred_mismatch = False  # Check if 'de' maps to they_words, and 'dem' to them_words
        if len(pred) == 0:
            continue

        for j, entity in enumerate(pred):

            target_indices.append(entity["index"])
            if entity["word"] not in (they_words + them_words):
                logger.info(
                    f"Swedish de/dem did not match corresponding English they/them/the/those for"
                    f"sentence: '{sentence}', and entity: '{entity}'"
                )
                contains_pred_mismatch = True
                continue

            begin_idx = entity["start"] + subtract_pad + offset
            end_idx = entity["end"] + subtract_pad + offset

            if entity["word"] in them_words and corrected_sv[i][j]["word"].lower() == "dem":
                wrong_word = "they" if entity["word"][0].islower() else "They"
            elif entity["word"] in they_words and corrected_sv[i][j]["word"].lower() == "de":
                wrong_word = "them" if entity["word"][0].islower() else "Them"
            else:
                contains_pred_mismatch = True
                continue

            sentence = (
                f"{sentence[:begin_idx]}~~{wrong_word}~~ **{entity['word']}**{sentence[end_idx:]}"
            )

            # Count all added text: words, strikethroughs (~~), spaces, parenthesis and bold (**).
            # Keep track of the offset in case of multiple de/dem errors in same sentence.
            offset += 9 + len(wrong_word)

        if not contains_pred_mismatch and (len(target_indices) == len(set(target_indices))):
            mistake_sens.append(sentence)
        offset = 0

    return mistake_sens


def create_reply_msg(df_post, pipes):
    # Header
    message = create_header(df_post)

    # Analys
    # message += insert_heading("Analys av kommentar")
    # message = add_paragraph(message)
    # message += wrongful_de_dem(df_post)
    message = add_paragraph(message)
    correct_sens = correct_sentence(df_post["pred"][0], df_post["sentences"][0])

    for sentence in correct_sens:
        message += add_quotation(sentence)
        message = add_paragraph(message)

    # message += create_analysis_legend()
    # message = add_paragraph(message)
    # message = add_horizontal_rule(message)
    # message = add_paragraph(message)

    correct_sens_en = correct_sentence_en(
        preds=[pipes[0]["pred_pipe"], pipes[1]["pred_pipe"]],
        correct_sens=pipes[1]["corrected_sentences"],
    )

    if len(correct_sens_en) > 0:
        # If all Swedish de/dem were matched against suitable English equivalent
        message += create_guide_en(df_post)

        for sentence in correct_sens_en:
            message = add_paragraph(message)
            message += add_quotation(sentence)
    else:
        # Guide
        message += create_guide(df_post)

    return message

