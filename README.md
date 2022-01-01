## SpråkpolisenBot

SpråkpolisenBot is a reddit bot that corrects Swedish language errors on the subreddit /r/sweden. More specifically it targets incorrect usages of 'de' and 'dem'. 

The bot uses a transformer neural network which was specifically trained to perform "token classification". The model was named **DeFormer**, as a word play on transformers and because it *deforms* instances of `de` and `dem`. DeFormer predicts each word piece (token) in a sentence in to one of three categories: `ord`, `DE` or `DEM`. Further details on the model powering SpråkpolisenBot can be found at [the DeFormer repository](https://github.com/Lauler/deformer).

## SpråkpolisenBot website

You can read more about SpråkpolisenBot at [https://lauler.github.io/sprakpolisen/](https://lauler.github.io/sprakpolisen/).