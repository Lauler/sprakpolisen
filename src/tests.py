from .data import filter_de_som, filter_dom

sens = [
    "Jag hatar de som är för sämst för de suger dem",
    "Jag har sett dem",
    "De som vill stanna kan se dem och dom.",
]
preds = [
    [
        {"start": 10, "end": 12, "word": "de"},
        {"start": 34, "end": 36, "word": "de"},
        {"start": 43, "end": 46, "word": "dem"},
    ],
    [{"start": 13, "end": 16, "word": "dem"}],
    [
        {"start": 0, "end": 2, "word": "De"},
        {"start": 26, "end": 29, "word": "dem"},
        {"start": 34, "end": 37, "word": "dom"},
    ],
]

filter_de_som(sens, preds)
filter_dom(sens, preds)
