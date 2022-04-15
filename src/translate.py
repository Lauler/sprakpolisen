import re
import logging
import matplotlib.pyplot as plt
from .utils import heatmap, annotate_heatmap
from .comment import match_case

logger = logging.getLogger(__name__)


def get_cross_attention(outputs, layer=-1):
    """
    Average cross attentions over all attention heads in a specific layer
    After transposing:
    Rows: encoder tokens (Swedish)
    Cols: decoder tokens (English)
    """

    return outputs.cross_attentions[layer].squeeze().mean(dim=0).t()


def correct_sen(preds, sentences):
    """
    Correct the Swedish sentence with mistake(s), and keep track of string indices of the
    corrected de/dem instances.
    """
    offset = 0
    correct_sens = []
    pred_list = []

    # Handle strange capitalizations like dE deM dEM
    de_pattern = "(?<!\w)[d][E](?!\w)"
    dem_pattern = "(?<!\w)[d][Ee][Mm](?!\w)"
    sentences = [re.sub(de_pattern, "de", sentence) for sentence in sentences]
    sentences = [re.sub(dem_pattern, "dem", sentence) for sentence in sentences]

    for pred, sentence in zip(preds, sentences):

        if len(pred) == 0:
            continue

        original_sentence = sentence

        new_entities = []
        for entity in pred:
            original_word, correct_word = match_case(original_sentence, entity)
            begin_idx = entity["start"] + offset
            end_idx = entity["end"] + offset
            sentence = f"{sentence[:begin_idx]}{correct_word}{sentence[end_idx:]}"

            # Keep track of the offset in case of multiple de/dem errors in same sentence.
            offset += len(correct_word) - len(original_word)

            new_entities.append(
                {"start": begin_idx, "end": entity["end"] + offset, "word": correct_word}
            )

        correct_sens.append(sentence)
        pred_list.append(new_entities)
        offset = 0

    out_dict = {"corrected_sentences": correct_sens, "pred_pipe": pred_list}

    return out_dict


def get_translation_tokens(swedish_pipe, model_translate, tokenizer_translate, device):
    """
    Get token ids and tokens in both Swedish and English to build a similar output to Huggingface's
    pipe()-function.

    Args: 
        swedish_pipe (dict): Takes the output from correct_sen function.
        {"corrected_sentences": ["De är...", ...], "pred_pipe": [[{'start': 0, 'end': 2, 'word': 'De'}], ...]}

    Returns:
        tuple: A Swedish and an English pipe output.
            Swedish:
            {"corrected_sentences": ["De är...", ...], 
            "token_sentences": [["_De", "_är", ...], ...]
            "pred_pipe": [[{'start': 0, 'end': 2, 'word': 'De'}], [{'start': 116, 'end': 118, 'word': 'de'}], ...],
            "inputs": {'input_ids': tensor([[  150,    18,  3577, ...]], device='cuda:0'), 
                'attention_mask': tensor([[1, 1, 1, ...]], device='cuda:0')}
            }
            English:
            {"corrected_sentences": ["They are...", ...], 
            "token_sentences": [['<pad>', '▁They', '▁are', ...], ...],
            "outputs": {'input_ids': tensor([[  150,    18,  3577, ...]], device='cuda:0')}
            }
    """

    tokenized_sens_sv = []
    tokens_sv = []
    for sentence in swedish_pipe["corrected_sentences"]:
        tokenized_sens_sv.append(
            tokenizer_translate(sentence, return_tensors="pt").to(device)
        )  # ids
        tokens_sv.append(
            tokenizer_translate.convert_ids_to_tokens(tokenized_sens_sv[-1].input_ids[0])
        )  # tokens

    tokenized_sens_en = []
    decoded_sens_en = []
    tokens_en = []
    model_translate.eval()

    for tok_sen in tokenized_sens_sv:
        tokenized_sens_en.append(model_translate.generate(**tok_sen))
        decoded_sens_en.append(
            tokenizer_translate.batch_decode(tokenized_sens_en[-1], skip_special_tokens=True)[0]
        )
        tokens_en.append(tokenizer_translate.convert_ids_to_tokens(tokenized_sens_en[-1][0]))

    swedish_pipe["token_sentences"] = tokens_sv
    swedish_pipe["outputs"] = tokenized_sens_sv

    english_pipe = {
        "corrected_sentences": decoded_sens_en,
        "token_sentences": tokens_en,
        "outputs": tokenized_sens_en,
    }

    return swedish_pipe, english_pipe


def translation_pipe(pipes):
    """
    Get the pipe output for every token in Swedish and English, e.g.
    pipe': [[{'start': 0, 'end': 4, 'index': 1, 'word': '<pad>'}, 
            {'start': 5, 'end': 9, 'index': 2, 'word': 'They'}, 
            {'start': 10, 'end': 13, 'index': 3, 'word': 'are'}, 
            ...]
    Add them to respective language's dict pipes.

    Note that English decoder tokens start with <pad>-token, whereas Swedish
    tokens from encoder don't start with <pad> (index correction is needed for 
    English later by subtracting with 5.) 
    """

    current_pipes = []
    for i, pipe in enumerate(pipes):
        for tok_sen in pipe["token_sentences"]:
            current_pipe = [
                {
                    "start": 0,
                    "end": len(tok_sen[0]) - 1,
                    "index": 1,
                    "word": tok_sen[0].strip("▁"),
                }
            ]
            for j, token in enumerate(tok_sen[1:]):
                token = token.replace("▁", " ")
                current_pipe.append(
                    {
                        "start": (current_pipe[j]["end"] + 1)
                        if token.startswith(" ")
                        else current_pipe[j]["end"],
                        "end": current_pipe[j]["end"] + (len(token) if token != "<unk>" else 1),
                        "index": j + 2,
                        "word": token.strip(),
                    }
                )

            current_pipes.append(current_pipe)

        pipes[i]["pipe"] = current_pipes
        current_pipes = []

    return pipes


def get_dedem_token_index(pipes):
    """
    Get token indices for de/dem predictions when using the translation tokenizer
    instead of KB-BERT's regular tokenizer.
    """

    sv_pipe = pipes[0]

    for i, preds in enumerate(sv_pipe["pred_pipe"]):
        for j, pred_token in enumerate(preds):
            for token in sv_pipe["pipe"][i]:

                if (pred_token["start"] == token["start"]) and (
                    pred_token["word"].lower() == token["word"].lower()
                ):
                    sv_pipe["pred_pipe"][i][j]["index"] = token["index"]

    return sv_pipe, pipes[1]


def decoder_pred_pipe(pipes, model_translate):
    """
    Match predicted de/dem corrections in Swedish sentence with 
    corresponding words in English sentence via attention scores.
    Hopefully the highest attention scores point to they/them/the/those 
    in the English sentence.
    """
    sv_pipe = pipes[0]
    en_pipe = pipes[1]

    model_translate.eval()
    en_pred_pipes = []

    for i, preds in enumerate(sv_pipe["pred_pipe"]):
        en_pred_pipe = []

        for j, pred_token in enumerate(preds):
            outputs = model_translate(
                input_ids=sv_pipe["outputs"][i].input_ids,
                decoder_input_ids=en_pipe["outputs"][i],
                output_hidden_states=True,
            )

            cross_attention = get_cross_attention(outputs, layer=0)

            # Index within sentence for incorrect de/dem token
            dedem_index_sv = pred_token["index"]
            # Which English token does de/dem attend to the most?
            index_en = cross_attention[dedem_index_sv - 1, 1:-1].argmax()

            # Add 1 to index because pytorch matrix index starts count from 0
            en_pred_pipe.append(en_pipe["pipe"][i][index_en + 1])

        en_pred_pipes.append(en_pred_pipe)

    en_pipe["pred_pipe"] = en_pred_pipes

    return sv_pipe, en_pipe


def visualize_attention(pipes, model_translate, sen_nr=0):

    outputs = model_translate(
        input_ids=pipes[0]["outputs"][sen_nr].input_ids,
        # attention_mask=tokenized_sens_sv[2].attention_mask,
        decoder_input_ids=pipes[1]["outputs"][sen_nr],
        output_hidden_states=True,
    )

    cross_attention = get_cross_attention(outputs, layer=0)

    encoder_text = pipes[0]["token_sentences"][sen_nr]
    decoder_text = pipes[1]["token_sentences"][sen_nr]

    fig, ax = plt.subplots(figsize=(10, 10))
    im, cbar = heatmap(
        cross_attention.cpu().detach()[:, :],
        row_labels=encoder_text[:],
        col_labels=decoder_text[:],
        ax=ax,
        cmap="YlGn",
        cbarlabel="Attentions",
    )

    texts = annotate_heatmap(im)

    fig.tight_layout()
    plt.show()


def translation_preprocess(df_post, model_translate, tokenizer_translate, device):
    sentence_pipe_sv = correct_sen(preds=df_post["pred"][0], sentences=df_post["sentences"][0])
    pipes = get_translation_tokens(sentence_pipe_sv, model_translate, tokenizer_translate, device)
    pipes = translation_pipe(pipes)
    pipes = get_dedem_token_index(pipes)
    pipes = decoder_pred_pipe(pipes, model_translate)

    return pipes
