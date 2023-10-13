# Learner Argument Corpus for Enthymeme Detection and Reconstruction

This repository contains the code to reproduce the data and results of the paper "Mind the Gap: Automated Corpus Creation for Enthymeme Detection and Reconstruction in Learner Arguments", as accepted to the EMNLP 2023 conference.

## Corpus Creation Process

Place the ICLEv3 corpus [1] in the folder "/data/ICLEv3". The folder "ICLEv3" should contain: metadata.csv, metadata.xlsx, and the subfolder "texts" with one json file per essay.
From "/code" run:
1. ``$python -m main -annotate_iclev3``
2. ``$python -m corpus_creation``
3. execute the notebook _corpus.ipynb_ ("/notebooks/corpus.ipynb")


## Corpus Annotations

Alternatively, we also provide the annotations of the created corpus but removed the text, since we are not allowed to provide the ICLEv3 data. However, the whole corpus can be recreated based on the essay IDs (column "File"), paragraph indices (column "Para No.") and the indices of the removed sentences (column "Position (Index)"). The corpus annotations can be found under "data/corpus/gap_corpus_annotations.csv".

Columns:

- File: essay ID
- Title: essay title
- Score: predicted essay quality score 
- Para No.: paragraph index of the argument within the essay; paragraphs splitted at "\n"
- Sentences: list of sentences of the original argument
- Paragraph: original argument text
- Para with Gap: argument text with created enthymematic gap
- Before Gap: argument text that preceeds the created gap
- Gap: removed ADU / enthymematic gap
- After Gap: argument text that follows the created gap
- ADU flow: sequence of predicted ADU types of the sentences
- ADU Target (Gap): predicted ADU type of the gap
- Position (Index): sentence index of the gap, sentences splitted using NLTK sentence splitter
- Sentence Score: quality contribution score of the gap sentence
- Rank Score: centrality score of the gap sentence
- labels: 0 for positive examples (actual gap), 1 for negative examples (random gap)
- Gap (Indicator): 1 for cases in which no gap was created


## Experiments

The code for our enthymeme detection and reconstruction experiments can be found in the "/notebooks" folder.

## Model weights

We will pubish the weights of all used models upon acceptance, since they exceed the allowable file size for the submission. 


## References

[1] Sylviane Granger, Maïté Dupont, Fanny Meunier, Hu759 bert Naets, and Magali Paquot. 2020. The International Corpus of Learner English. Version 3. Presses universitaires de Louvain, Louvain-la-Neuve. (http://hdl.handle.net/2078.1/229877)
