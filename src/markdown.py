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
        wrongful_msg = f"**{dem_nr}** felaktiga användningar av **`dem`**"
    elif de_nr > 0 and dem_nr == 0:
        wrongful_msg = f"**{de_nr}** felaktiga användningar av **`de`**"
    elif de_nr > 0 and dem_nr > 0:
        wrongful_msg = (
            f"**{de_nr}** felaktiga användningar av **`de`** "
            f"samt **{dem_nr}** felaktiga användningar av **`dem`**"
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

    message = (
        f"Efter en analys av inlägget har mitt neurala nätverk upptäckt {wrongful_msg}. "
        f"SpråkpolisenBot föreslår följande ändringar:"
    )

    return message


def create_analysis_legend():
    message = (
        f"~~ord~~: Överstruket ord indikerar felaktiv användning av ~~de~~ eller ~~dem~~`.  \n"
        f"**ord**: Fetstilt **de/dem** är SpråkpolisenBots förslag till korrigering.  \n"
        f"**(##.##%)**: Siffror inom parentes indikerar hur pass säker modellen är på sin prediktion."
    )

    return message


def create_header(df_post):

    if df_post["nr_mistakes"][0] == 1:
        message = (
            f'Tjenixen, /u/SprakpolisenBot här {add_emoji("police")}. Jag är en båt som '
            f"tränats till att kunna skilja mellan korrekt och felaktigt bruk av `de` och `dem` i svensk text. "
        )
    if df_post["nr_mistakes"][0] == 2:
        message = (
            f'/u/SprakpolisenBot här {add_emoji("police")}{add_emoji("car")}. '
            f"Vi utför för närvarande slumpmässiga språkkontroller av kommentarer på /r/sweden. "
            f"Ovanstående inlägg överskred den tillåtna gränsen för felaktiga `de/dem`-användningar. "
            f"Vi rekommenderar användare som vill undvika att fastna i framtida kontroller att "
            f"ta del av analysen och guiden som bifogas nedan."
        )
    if df_post["nr_mistakes"][0] >= 3:
        message = (
            f'Stopp {add_emoji("car")}{add_emoji("siren")}! '
            f'Du har blivit anhållen av /u/SprakpolisenBot {add_emoji("police")} '
            f"på sannolika skäl misstänkt för brott mot det svenska skriftspråket."
        )

    return message


def create_guide(df_post):
    guide_message = (
        f"En guide med tips och strategier för att skilja mellan `de` och `dem` finnes "
        f"på [Språkpolisens hemsida](https://lauler.github.io/sprakpolisen/guide.html). "
        f"De tillämpningsmässigt enklaste och minst tidskrävande tipsen har listats först. "
        f"Ett interaktivt demo där användare själva kan skriva in meningar och få dem "
        f"rättade finns också tillgänglig. Instruktioner för hur demot kan nås och användas "
        f"[hittas här](https://lauler.github.io/sprakpolisen/interactive.html)."
    )

    if df_post["nr_mistakes_dem"][0] >= 2:
        message = (
            f"Visste du att `de` är cirka 10 gånger vanligare än `dem` i svensk text? "
            f"Om du är osäker kring vilket som är rätt är det alltså statistiskt sett säkrast "
            f"att ***alltid gissa på `de`.***"
        )
        message = add_paragraph(message)
        message += guide_message
    else:
        message = guide_message

    return message


def create_footer():
    message = (
        f"^([Om SpråkpolisenBot](https://lauler.github.io/sprakpolisen)) | "
        f"^([Källkod](https://github.com/Lauler/sprakpolisen)) | "
        f"^([Vanliga frågor](https://lauler.github.io/sprakpolisen/faq.html)) | "
        f"^([Feedback](https://lauler.github.io/sprakpolisen/contact.html)) | "
        f"^([Interaktivt demo](https://lauler.github.io/sprakpolisen/demo.html)) "
    )

    return message
