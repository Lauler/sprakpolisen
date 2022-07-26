import re


def add_quotation(sentence):
    return "> " + sentence


def add_paragraph(text):
    return text + " \n\n"


def add_emoji(type):
    """
    Add Unicode emojis.
    """

    if type == "police":
        # https://unicode-table.com/en/1F46E/ 👮
        html_code = "&#128110;"
    elif type == "car":
        # https://unicode-table.com/en/1F693/ 🚓
        html_code = "&#128659;"
    elif type == "siren":
        # https://unicode-table.com/en/1F6A8/ 🚨
        html_code = "&#128680;"

    return html_code


def add_horizontal_rule(text):
    return text + "---"


def insert_heading(header):
    return f"## {header}"


def remove_emoji(text):
    # https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python

    emoji_pattern = re.compile("[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)

    return emoji_pattern.sub("", text)


def reverse_replace(text, old, new, n):
    """
    Replace n occurences of 'old' with 'new' starting from the right side of the text.
    """
    split_sen = text.rsplit(old, n)
    return new.join(split_sen)


def wrongful_de_dem(df_post):
    de_nr = df_post["nr_mistakes_de"].iloc[0]
    dem_nr = df_post["nr_mistakes_dem"].iloc[0]

    if de_nr == 0 and dem_nr > 0:
        wrongful_msg = f"**{dem_nr}** felaktiga användningar av `dem`"
    elif de_nr > 0 and dem_nr == 0:
        wrongful_msg = f"**{de_nr}** felaktiga användningar av `de`"
    elif de_nr > 0 and dem_nr > 0:
        wrongful_msg = (
            f"**{de_nr}** felaktiga användningar av `de` "
            f"samt **{dem_nr}** felaktiga användningar av `dem`"
        )

    if (
        (de_nr == 1 and dem_nr == 0)
        or (de_nr == 0 and dem_nr == 1)
        or (de_nr == 1 and dem_nr == 1)
    ):
        wrongful_msg = wrongful_msg.replace("felaktiga", "felaktig")
        wrongful_msg = wrongful_msg.replace("användningar", "användning")

    if de_nr == 1 and dem_nr > 1:
        wrongful_msg = wrongful_msg.replace("felaktiga", "felaktig", 1)
        wrongful_msg = wrongful_msg.replace("användningar", "användning", 1)

    if de_nr > 1 and dem_nr == 1:
        wrongful_msg = reverse_replace(wrongful_msg, "felaktiga", "felaktig", 1)
        wrongful_msg = reverse_replace(wrongful_msg, "användningar", "användning", 1)

    message = f"Efter en analys har mitt neurala nätverk upptäckt {wrongful_msg}."

    return message


def create_analysis_legend():
    message = (
        # f"~~ord~~: Överstruket ord indikerar felaktig användning av ~~de~~ eller ~~dem~~.  \n"
        # f"Fetstilt **de/dem** är SpråkpolisenBots förslag till korrigering."
        # f"**(##.##%)**: Siffror inom parentes indikerar hur pass säker modellen är på sin prediktion."
    )

    return message


def create_header(df_post):

    if df_post["nr_mistakes"][0] <= 2:
        message = (
            f'Tjenixen, SpråkpolisenBot här {add_emoji("police")}. Jag är en bot som '
            f"skiljer på `de` och `dem`. "
        )
    if df_post["nr_mistakes"][0] >= 3:
        message = (
            f'Stopp {add_emoji("car")}{add_emoji("siren")}! '
            f'Du har blivit gripen av SpråkpolisenBot {add_emoji("police")} '
            f"på sannolika skäl misstänkt för brott mot det svenska skriftspråket. "
        )

    # if df_post["nr_mistakes"][0] == 2:
    #     message = (
    #         f'SpråkpolisenBot här {add_emoji("police")}{add_emoji("car")}. '
    #         f"Vi utför för närvarande slumpmässiga språkkontroller av kommentarer på Sweddit. "
    #         f"Ovanstående inlägg överskred den tillåtna gränsen för felaktiga `de/dem`-användningar. "
    #         f"Vi rekommenderar användare som vill undvika att fastna i framtida kontroller att "
    #         f"ta del av analysen och guiden som bifogas nedan."
    #     )

    return message


def create_guide(df_post):
    guide_message = (
        f"En guide med tips för att skilja på `de` och `dem` finnes "
        f"på [Språkpolisens hemsida](https://lauler.github.io/sprakpolisen/guide.html). "
        # f"En interaktiv demo där användare själva kan skriva in meningar och få dem "
        # f"rättade [finns här](https://lauler.github.io/sprakpolisen/demo.html)."
    )

    message = ""

    if df_post["nr_mistakes_dem"][0] >= 2:
        added_message = (
            f"Visste du att `de` är cirka 10 gånger vanligare än `dem` i svensk text? "
            f"Om du är osäker kring vilket som är rätt är det alltså statistiskt sett säkrast "
            f"att ***alltid gissa på `de`.***"
        )
        message += add_paragraph(added_message)

    if df_post["sentences"].apply(
        lambda sens: any([bool(re.search("[Dd]em flesta", sen)) for sen in sens])
    )[0]:
        added_message = (
            f"Visste du att det aldrig kan heta ~~dem flesta~~ på svenska? **De flesta** "
            f"är den enda korrekta formen av uttrycket."
        )
        message += add_paragraph(added_message)

    for word in ["andra", "värsta", "bästa", "sämsta", "första"]:
        if df_post["sentences"].apply(
            lambda sens: any([bool(re.search(f"[Dd]em {word}", sen)) for sen in sens])
        )[0]:
            added_message = (
                f"Visste du att det inte kan heta ~~dem {word}~~? **De {word}** "
                f"är den korrekta formen. När `de` används i en betydelse "
                f"som motsvarar engelskans **the**, ska det alltid vara `de` på svenska."
            )
            message += add_paragraph(added_message)

    message += guide_message

    return message


def create_guide_en(df_post):
    """
    Guide if English sentences are successfully translated and included in message.
    """

    guide_message = (
        f"[Tips](https://lauler.github.io/sprakpolisen/guide.html): Använd engelskan till hjälp. "
        f"Om **them** passar bäst ska det vara `dem` på svenska. "
        f"Om **they/those/the** eller något annat passar bättre ska det vara `de`."
    )

    message = ""

    if df_post["sentences"].apply(
        lambda sens: any([bool(re.search("[Dd]em flesta", sen)) for sen in sens])
    )[0]:
        added_message = (
            f"Visste du att det aldrig kan heta ~~dem flesta~~ på svenska? **De flesta** "
            f"är den enda korrekta formen av uttrycket."
        )
        message += add_paragraph(added_message)

    for word in ["andra", "värsta", "bästa", "sämsta", "första"]:
        if df_post["sentences"].apply(
            lambda sens: any([bool(re.search(f"[Dd]em {word}", sen)) for sen in sens])
        )[0]:
            added_message = (
                f"Visste du att det inte kan heta ~~dem {word}~~? **De {word}** "
                f"är den korrekta formen. När `de` används i en betydelse "
                f"som motsvarar engelskans **the**, ska det alltid vara `de` på svenska."
            )
            message += add_paragraph(added_message)

    message += guide_message

    return message


def create_footer():
    # &nbsp; is space character, needed for compatibility with old reddit.
    message = (
        f"^[Om&nbsp;SpråkpolisenBot](https://lauler.github.io/sprakpolisen) | "
        f"^[Källkod](https://github.com/Lauler/sprakpolisen) | "
        f"^[Vanliga&nbsp;frågor](https://lauler.github.io/sprakpolisen/faq.html) | "
        f"^[Feedback](https://lauler.github.io/sprakpolisen/contact.html) | "
        f"^[Interaktiv&nbsp;demo](https://lauler.github.io/sprakpolisen/demo.html) "
    )

    return message
