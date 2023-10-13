import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import time
from builtins import set

from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from functools import partial
import language_tool_python
from utils import *
import pandas as pd
import torch
import itertools
import pickle
from models import ADUClassifier
from predict import predict_ADU_Non, predict_ADU, embedding_encoding
import warnings
import spacy
from collections import OrderedDict
from statistics import mean, median
from sentence_transformers import util
import jsonpickle
import regex
from transformers import pipeline
import argparse


nlp = spacy.load('en_core_web_sm')
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transition_list = ['also', 'again', 'as well as', 'besides', 'coupled with', 'furthermore', 'in addition', 'likewise',
                   'moreover', 'similarly', 'accordingly', 'as a result', 'consequently', 'for this reason',
                   'for this purpose', 'hence', 'otherwise', 'so then', 'subsequently', 'therefore', 'thus',
                   'thereupon', 'wherefore', 'contrast', 'by the same token', 'conversely', 'instead', 'likewise',
                   'on one hand', 'on the other hand', 'on the contrary', 'rather', 'similarly', 'yet', 'but',
                   'however',
                   'still', 'nevertheless', 'in contrast', 'here', 'there', 'over there', 'beyond', 'nearly',
                   'opposite', 'under', 'above',
                   'to the left', 'to the right', 'in the distance', 'by the way', 'incidentally ', 'above all',
                   'chiefly', 'with attention to', 'especially',
                   'particularly', 'singularly', 'aside from', 'barring', 'beside', 'except', 'excepting', 'excluding',
                   'exclusive of', 'other than',
                   'outside of', 'save', 'chiefly', 'especially', 'for instance', 'in particular', 'markedly', 'namely',
                   'particularly', 'including', 'specifically', 'such as', 'as a rule', 'as usual', 'for the most part',
                   'generally', 'generally speaking', 'ordinarily',
                   'usually', 'for example', 'for instance', 'for one thing', 'as an illustration', 'illustrated with',
                   'as an example', 'in this case',
                   'comparatively', 'coupled with', 'correspondingly', 'identically', 'likewise', 'similar', 'moreover',
                   'together with', 'in essence', 'in other words', 'namely', 'that is', 'that is to say', 'in short',
                   'in brief', 'to put it differently',
                   'at first', 'first of all', 'to begin with', 'in the first place', 'at the same time',
                   'for now', 'for the time being', 'the next step', 'in time', 'in turn', 'later on',
                   'meanwhile', 'next', 'then', 'soon', 'the meantime', 'later', 'while', 'earlier',
                   'simultaneously', 'afterward', 'in conclusion', 'with this in mind',
                   'after all', 'all in all', 'all things considered', 'briefly', 'by and large', 'in any case',
                   'in any event',
                   'in brief', 'in conclusion', 'on the whole', 'in short', 'in summary', 'in the final analysis',
                   'in the long run', 'on balance', 'to sum up', 'to summarize', 'finally']


def set_adu(essays):
    """
    Predicts ADU for each sentence. ADUs are represented as adu_flow list attribute for each paragraph.

    :param essays: List of essays
    :return: Pandas dataframe (examples x features)
    """
    all_rows = []
    for essay in essays:
        for para in essay.paragraphs:
            for sent in para.sentences:
                all_rows.append([essay.file, sent])
    df_essays = pd.DataFrame(all_rows, columns=['file', 'Text'])
    CLASS_NAMES = ['Non-ADU', 'ADU']
    model = ADUClassifier(len(CLASS_NAMES))
    model.load_state_dict(torch.load("../models/ADU_NonADU_NoFeatures_params.pt", map_location=device))

    result = predict_ADU_Non(model, df_essays)



    CLASS_NAMES = ['MajorClaim', 'Claim', 'Premise']
    model = ADUClassifier(len(CLASS_NAMES))

    model.load_state_dict(torch.load("../models/ADU_NoFeatures_params.pt", map_location=device))

    result = predict_ADU(model, result)

    index = 0
    for essay in essays:
        for para in essay.paragraphs:
            for i in range(len(para.sentences)):
                para.adu_flow.append(result["Ann_Adu"].iloc[index])
                index += 1


def set_entities(essays):
    """
    Sets para.entities for each essay's paragraph using spaCy.

    :param essays: List of essays
    :return:
    """
    for essay in essays:
        for para in essay.paragraphs:
            para.entities = []
            for sent in para.sentences:
                para.entities.append([ent.label_ for ent in nlp(sent).ents])


def set_para_flow(essays):
    """
    Sets paragraph function flow based on the heuristic by Persing, I., Davis, A., & Ng, V. (2010, October).
    Modeling organization in student essays. In Proceedings of the 2010 conference on empirical methods in natural language processing (pp. 229-239).


    :param essays: List of essays
    :return:
    """
    for essay in essays:
        essay.paragraph_labels = []
        for i, para in enumerate(essay.paragraphs):
            labels = {
                'Introduction': 0,
                'Body': 0,
                'Conclusion': 0,
                'Rebuttal': 0
            }
            for j, sent in enumerate(para.sentences):
                s_label = para.sentence_labels[j]
                if s_label == 'Thesis' or s_label == 'Prompt':
                    labels['Introduction'] += 1
                    labels['Conclusion'] += 1
                if s_label == 'MainIdea' and j < 3:
                    labels['Body'] += 1
                    labels['Conclusion'] += 1
                    labels['Rebuttal'] += 1
                if s_label == 'MainIdea' and j > (len(para.sentences) - 4):
                    labels['Introduction'] += 1
                    labels['Body'] += 1
                if s_label == 'Elaboration':
                    labels['Introduction'] += 1
                    labels['Body'] += 1
                if s_label == 'Support':
                    labels['Body'] += 1
                if s_label == 'Conclusion' or s_label == 'Suggestion':
                    labels['Body'] += 1
                    labels['Conclusion'] += 1
                if s_label == 'Rebuttal' or s_label == 'Solution':
                    labels['Body'] += 1
                    labels['Rebuttal'] += 1
            if i == 0:
                labels['Introduction'] += 1
            if i == (len(essay.paragraphs) - 1):
                labels['Conclusion'] += 1
            if 0 < i < (len(essay.paragraphs) - 1):
                labels['Body'] += 1
                labels['Rebuttal'] += 1
            essay.paragraph_labels.append(max(labels, key=labels.get))


def set_flow_variations(essays):
    """
    Sets all adu flows (described in Wachsmuth, H., Al Khatib, K., & Stein, B. (2016, December)) derived from adu_flow.

    :param essays: List of essays
    :return:
    """
    for essay in essays:
        for para in essay.paragraphs:
            para.adu_flow_wo_none = [x for x in para.adu_flow if x != 'None']
            para.adu_changes_only = [k for k, g in itertools.groupby(para.adu_flow)]
            para.adu_changes_wo_none = [x for x in para.adu_changes_only if x != 'None']
            para.adu_wo_none_changes = [k for k, g in itertools.groupby(para.adu_flow_wo_none)]


def set_sentence_labels(essays):
    """
    Sets the sentence labels for each paragraph based on the heuristic by Persing, I., Davis, A., & Ng, V. (2010, October).
    Modeling organization in student essays. In Proceedings of the 2010 conference on empirical methods in natural language processing (pp. 229-239).


    :param essays: List of essays
    :return:
    """
    pos_tags = ['NN', 'VBN', 'JJ', 'JJS', 'NNS', 'NNP', 'NNPS', 'JJR', 'VB', 'VBD', 'VBG', 'VBP', 'VBZ', 'RB', 'RBR',
                'RBS']
    for essay in essays:
        title = essay.title.lower()
        index = 0
        title_list_A = [word for (word, pos) in pos_tag(word_tokenize(title)) if pos[:2] in pos_tags]
        for para in essay.paragraphs:
            para.sentence_labels = []
            for sent in para.sentences:

                labels = OrderedDict({
                    'Elaboration': 0,
                    'Prompt': 0,
                    'Transition': 0,
                    'Thesis': 0,
                    'MainIdea': 0,
                    'Support': 0,
                    'Conclusion': 0,
                    'Rebuttal': 0,
                    'Solution': 0,
                    'Suggestion': 0
                })
                s_lower = sent.lower()
                for words in ["they", "them", "my", "he", "she"]:
                    if re.search(r'\b' + words + r'\b', s_lower):
                        labels["Elaboration"] = labels["Elaboration"] + 1
                if index == 0:
                    labels["Prompt"] = labels["Prompt"] + 1
                sent_list_B = [word for (word, pos) in pos_tag(word_tokenize(s_lower)) if pos[:2] in pos_tags]
                common_C = list(set(sent_list_B).intersection(title_list_A))
                if len(sent_list_B) > 0:
                    labels["Prompt"] = labels["Prompt"] + (5 / 2) * len(common_C) / len(sent_list_B)

                if '?' in s_lower:
                    labels["Transition"] = labels["Transition"] + 1
                for words in transition_list:
                    if re.search(r'\b' + words + r'\b', s_lower):
                        labels["Transition"] = labels["Transition"] + 1
                        break
                for words in ["agree", "disagree", "think", "opinion"]:
                    if re.search(r'\b' + words + r'\b', s_lower):
                        labels["Thesis"] = labels["Thesis"] + 1
                for words in ["firstly", "secondly", "another", "aspect"]:
                    if re.search(r'\b' + words + r'\b', s_lower):
                        labels["MainIdea"] = labels["MainIdea"] + 1
                for words in ["example", "instance"]:
                    if re.search(r'\b' + words + r'\b', s_lower):
                        labels["Support"] = labels["Support"] + 1
                for words in ["conclusion", "conclude", "therefore", "sum"]:
                    if re.search(r'\b' + words + r'\b', s_lower):
                        labels["Conclusion"] = labels["Conclusion"] + 1
                for words in ["however", "but", "argue"]:
                    if re.search(r'\b' + words + r'\b', s_lower):
                        labels["Rebuttal"] = labels["Rebuttal"] + 1
                for words in ["solve", "solved", "solution"]:
                    if re.search(r'\b' + words + r'\b', s_lower):
                        labels["Solution"] = labels["Solution"] + 1
                for words in ["should", "let", "must", "ought"]:
                    if re.search(r'\b' + words + r'\b', s_lower):
                        labels["Suggestion"] = labels["Suggestion"] + 1
                index += 1
                para.sentence_labels.append(max(labels, key=labels.get))


def identity(x):
    return x


def adu_flow_features(essays, train=True):
    """
    Calculate ADU flow features.

    :param essays: List of essays
    :param train: Boolean indicator if fitted vectorizer should be stored for later prediction
    :return: Pandas dataframe (examples x features)
    """
    X_adu_flows = []
    X_adu_flow_wo_none = []
    X_adu_changes_only = []
    X_adu_changes_wo_none = []
    X_adu_wo_none_changes = []

    for essay in essays:
        essay_adu_flows = []
        essay_adu_flow_wo_none = []
        essay_adu_changes_only = []
        essay_adu_changes_wo_none = []
        essay_adu_wo_none_changes = []

        for para in essay.paragraphs:
            essay_adu_flows.append(tuple(para.adu_flow))
            essay_adu_flow_wo_none.append(tuple(para.adu_flow_wo_none))
            essay_adu_changes_only.append(tuple(para.adu_changes_only))
            essay_adu_changes_wo_none.append(tuple(para.adu_changes_wo_none))
            essay_adu_wo_none_changes.append(tuple(para.adu_wo_none_changes))
        X_adu_flows.append(essay_adu_flows)
        X_adu_flow_wo_none.append(essay_adu_flow_wo_none)
        X_adu_changes_only.append(essay_adu_changes_only)
        X_adu_changes_wo_none.append(essay_adu_changes_wo_none)
        X_adu_wo_none_changes.append(essay_adu_wo_none_changes)

    if train:
        vectorizer = CountVectorizer(analyzer=identity, min_df=0.01)
        X_adu_flows = vectorizer.fit_transform(X_adu_flows)
        # print(X_adu_flows.todense(), type(X_adu_flows.todense()))

        X_adu_flows = pd.DataFrame(X_adu_flows.todense(), columns=vectorizer.get_feature_names_out())
        pickle.dump(vectorizer, open("../models/aduflow_vec.pkl", "wb"))
        X_adu_flow_wo_none = vectorizer.fit_transform(X_adu_flow_wo_none)
        X_adu_flow_wo_none = pd.DataFrame(X_adu_flow_wo_none.todense(), columns=vectorizer.get_feature_names_out())
        pickle.dump(vectorizer, open("../models/adu_flow_wo_none_vec.pkl", "wb"))
        X_adu_changes_only = vectorizer.fit_transform(X_adu_changes_only)
        X_adu_changes_only = pd.DataFrame(X_adu_changes_only.todense(), columns=vectorizer.get_feature_names_out())
        pickle.dump(vectorizer, open("../models/adu_changes_only_vec.pkl", "wb"))
        X_adu_changes_wo_none = vectorizer.fit_transform(X_adu_changes_wo_none)
        X_adu_changes_wo_none = pd.DataFrame(X_adu_changes_wo_none.todense(),
                                             columns=vectorizer.get_feature_names_out())
        pickle.dump(vectorizer, open("../models/adu_changes_wo_none_vec.pkl", "wb"))
        X_adu_wo_none_changes = vectorizer.fit_transform(X_adu_wo_none_changes)
        X_adu_wo_none_changes = pd.DataFrame(X_adu_wo_none_changes.todense(),
                                             columns=vectorizer.get_feature_names_out())
        pickle.dump(vectorizer, open("../models/adu_wo_none_changes_vec.pkl", "wb"))
    else:
        vectorizer = pickle.load(open("../models/aduflow_vec.pkl", 'rb'))
        X_adu_flows = vectorizer.transform(X_adu_flows)
        X_adu_flows = pd.DataFrame(X_adu_flows.todense(), columns=vectorizer.get_feature_names_out())
        vectorizer = pickle.load(open("../models/adu_flow_wo_none_vec.pkl", 'rb'))
        X_adu_flow_wo_none = vectorizer.transform(X_adu_flow_wo_none)
        X_adu_flow_wo_none = pd.DataFrame(X_adu_flow_wo_none.todense(), columns=vectorizer.get_feature_names_out())
        vectorizer = pickle.load(open("../models/adu_changes_only_vec.pkl", 'rb'))
        X_adu_changes_only = vectorizer.transform(X_adu_changes_only)
        X_adu_changes_only = pd.DataFrame(X_adu_changes_only.todense(), columns=vectorizer.get_feature_names_out())
        vectorizer = pickle.load(open("../models/adu_changes_wo_none_vec.pkl", 'rb'))
        X_adu_changes_wo_none = vectorizer.transform(X_adu_changes_wo_none)
        X_adu_changes_wo_none = pd.DataFrame(X_adu_changes_wo_none.todense(),
                                             columns=vectorizer.get_feature_names_out())
        vectorizer = pickle.load(open("../models/adu_wo_none_changes_vec.pkl", 'rb'))
        X_adu_wo_none_changes = vectorizer.transform(X_adu_wo_none_changes)
        X_adu_wo_none_changes = pd.DataFrame(X_adu_wo_none_changes.todense(),
                                             columns=vectorizer.get_feature_names_out())

    return pd.concat(
        [X_adu_flows, X_adu_flow_wo_none, X_adu_changes_only, X_adu_changes_wo_none, X_adu_wo_none_changes], axis=1)


def find_ngrams(input_list, n=3):
    return zip(*[input_list[i:] for i in range(n)])


def adu_ngram_features(essays, n=(1, 3), train=True):
    """
    Calculate ADU n-gram features.


    :param essays: List of essays
    :param n: tuple (1-3) grams
    :param train: Boolean indicator if fitted vectorizer should be stored for later prediction
    :return: Pandas dataframe (examples x features)
    """
    min, max = n
    features = []
    essays_adus = []

    for essay in essays:
        essay_adu = []
        for para in essay.paragraphs:
            essay_adu.extend(para.adu_flow)
        essays_adus.append(essay_adu)
    # print(essays_adus)
    for i in range(min, max + 1):
        if train:
            vectorizer = CountVectorizer(analyzer=partial(find_ngrams, n=i), min_df=0)
            vectorizer.fit(essays_adus)
            pickle.dump(vectorizer, open("../models/" + str(i) + "gram_vec.pkl", "wb"))


        else:
            vectorizer = pickle.load(open("../models/" + str(i) + "gram_vec.pkl", "rb"))

        features.append((vectorizer.transform(essays_adus), vectorizer.get_feature_names_out()))


    return pd.concat([pd.DataFrame(X[0].todense(), columns=X[1]) for X in features], axis=1)


def pos_ngram_features(essays, n=(1, 5), train=True):
    """
    Calculate POS 1-5 gram features.

    :param essays: List of essays
    :param n: tuple (1-5) grams
    :param train: Boolean indicator if fitted vectorizer should be stored for later prediction
    :return: Pandas dataframe (examples x features)
    """
    min, max = n
    features = []
    essays_pos = []
    for essay in essays:
        essay_pos = []
        # essay_text = word_tokenize(essay.text)
        # essay_pos.extend([pos[1] for pos in pos_tag(essay_text)])

        for para in essay.paragraphs:
            for pos in para.pos_tags:
                essay_pos.extend(pos)
        essays_pos.append(essay_pos)

    for i in range(min, max + 1):
        if train:
            vectorizer = CountVectorizer(analyzer=partial(find_ngrams, n=i), min_df=0.01)
            vectorizer.fit(essays_pos)
            pickle.dump(vectorizer, open("../models/" + str(i) + "pos_gram_vec.pkl", "wb"))
        else:
            vectorizer = pickle.load(open("../models/" + str(i) + "pos_gram_vec.pkl", "rb"))

        features.append((vectorizer.transform(essays_pos), vectorizer.get_feature_names_out()))

    return pd.concat([pd.DataFrame(X[0].todense(), columns=X[1]) for X in features], axis=1)


def adu_composition_feature(essays, train=True):
    """
    Calculate ADU composition feature described in Wachsmuth, H., Al Khatib, K., & Stein, B. (2016, December).


    :param essays: List of essays
    :param train: Boolean indicator if fitted vectorizer should be stored for later prediction
    :return: Pandas dataframe (examples x features)
    """
    composition = []
    labels = ["Premise", "Claim", "MajorClaim", "None"]
    columns_prefix = ["P_0_", "P_1_", "P_2_", "P_>2_", "P_min_", "P_max_", "P_mean_", "P_median_"]
    columns = []
    for label in labels:
        for p in columns_prefix:
            columns.append(p + label)

    for essay in essays:
        comp_dict = {
            # 0 1 2 >2
            "Premise": [0] * 8,
            "Claim": [0] * 8,
            "MajorClaim": [0] * 8,
            "None": [0] * 8
        }
        adu_dict = {
            "Premise": [0] * len(essay.paragraphs),
            "Claim": [0] * len(essay.paragraphs),
            "MajorClaim": [0] * len(essay.paragraphs),
            "None": [0] * len(essay.paragraphs)
        }
        for index, para in enumerate(essay.paragraphs):
            for adu in para.adu_flow:
                adu_dict[adu][index] += 1

            for adu in comp_dict:
                if adu_dict[adu][index] == 0:
                    comp_dict[adu][0] += 1
                if adu_dict[adu][index] == 1:
                    comp_dict[adu][1] += 1
                if adu_dict[adu][index] == 2:
                    comp_dict[adu][2] += 1
                if adu_dict[adu][index] > 2:
                    comp_dict[adu][3] += 1

        for k, v in adu_dict.items():
            # print(k, v)
            comp_dict[k] = [x / len(essay.paragraphs) for x in comp_dict[k]]
            comp_dict[k][4] = min(v)
            comp_dict[k][5] = max(v)
            comp_dict[k][6] = mean(v)
            comp_dict[k][7] = median(v)
        composition.append(sum(comp_dict.values(), []))

    return pd.DataFrame(composition, columns=columns)


def token_ngrams(essays, train=True):
    """
    Calculate token n-gram features (1-3)

    :param essays: List of essays
    :param train: Boolean indicator if fitted vectorizer should be stored for later prediction
    :return: Pandas dataframe (examples x features)
    """
    essays_texts = []
    for essay in essays:
        text = [essay.title]
        for para in essay.paragraphs:
            text.extend(para.sentences)
        essays_texts.append(' '.join(text))
    if train:
        vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=0.05)
        vectorizer.fit(essays_texts)
        pickle.dump(vectorizer, open("../models/token_ngrams_vec.pkl", "wb"))
    else:
        vectorizer = pickle.load(open("../models/token_ngrams_vec.pkl", "rb"))
    X = vectorizer.transform(essays_texts)
    columns = vectorizer.get_feature_names_out()
    return pd.DataFrame(X.todense(), columns=columns)


def length_stats(essays, train=True):
    """
    Calculate length based features

    :param essays: List of essays
    :param train:
    :return: Pandas dataframe (examples x features)
    """
    features = []
    for essay in essays:
        feature = []
        paras_lengths = []
        feature.append(len(essay.paragraphs))
        feature.append(len(essay))
        for para in essay.paragraphs:
            paras_lengths.append(len(para.sentences))
        feature.append(max(paras_lengths))
        feature.append(min(paras_lengths))
        feature.append(sum(paras_lengths))
        feature.append(mean(paras_lengths))

        features.append(feature)
    return pd.DataFrame(features, columns=["Num Para", "Num Tokens", "Max Para Sent", "Min Para Sent", "Num Sent",
                                           "Avg. Sent per Para"])


def f(x, values, length):
    x = np.asanyarray(x)
    k = len(values)
    intervals = [0]
    x_0 = length / k
    for i in range(k):
        intervals.append(x_0)
        x_0 += length / k
    intervals[-1] = (length + 0.01)

    y = []
    for i in range(1, len(intervals)):
        if np.ndim(x) == 0:
            if intervals[i - 1] <= x < intervals[i]:
                y.append(values[i - 1])
        else:
            for x_ in x:
                if intervals[i - 1] <= x_ < intervals[i]:
                    y.append(values[i - 1])
    return y


def length_normalize(values, length=16):
    """
    Map values to a sequence length of length=16

    :param values: List of values
    :param length: Normalized length. Defaults to 16
    :return:
    """
    y = f(np.arange(0, length, 1), values, length)
    return y


def function_flow(essays, train=True):
    """
    Calculate function flow feature based on paragraph labels.


    :param essays: List of essays
    :param train:
    :return: Pandas dataframe (examples x features)
    """
    features = []
    length = 16
    for essay in essays:
        features.append(length_normalize(list(map(func_to_value, essay.paragraph_labels)), length=length))


    columns = ['Function Flow_' + str(i) for i in range(1, length + 1)]
    return pd.DataFrame(features, columns=columns)


def sentiment_flow(essays, train=True):
    """
    Calculate normalized paragraph sentiment flow feature.


    :param essays: List of essays
    :param train:
    :return: Pandas dataframe (examples x features)
    """
    features = []
    length = 16
    for essay in essays:
        avg_sentiments = []
        for para in essay.paragraphs:
            if para.sentiment:
                avg_sentiments.append(mean(para.sentiment))
        features.append(length_normalize(avg_sentiments, length=length))

    columns = ['Sentiment Flow_' + str(i) for i in range(1, length + 1)]
    return pd.DataFrame(features, columns=columns)


def similarity_flow(essays, train=True):
    """
    Calculate normalized paragraph paragraph to prompt flow feature.


    :param essays: List of essays
    :param train:
    :return: Pandas dataframe (examples x features)
    """
    features = []
    length = 16
    for essay in essays:

        flow = []

        embeddings = essay.title_embeddings
        for para in essay.paragraphs:
                  flow.append(util.cos_sim(embeddings, para.paragraph_embeddings).item())

        features.append(length_normalize(flow))
    columns = ['Similarity Flow_' + str(i) for i in range(1, length + 1)]
    return pd.DataFrame(features, columns=columns)


def func_to_value(label):
    """
    Maps function label to value.
    Introduction/Conclusion is set to 0.5
    Body is set to 1
    Rebuttal is set to 0

    :param label: String representing paragraph function
    :return: Numeric value 0, 0.5, 1
    """
    if label == 'Introduction' or label == 'Conclusion':
        return 0.5
    if label == 'Body':
        return 1
    if label == 'Rebuttal':
        return 0


def entities(essays, train=True):
    """
    Occurences of named entities (feature) based on spaCy

    :param essays: List of essays
    :param train: Boolean indicator if fitted vectorizer should be stored for later prediction
    :return: Pandas dataframe (examples x features)
    """
    features = []
    entities = []
    for essay in essays:
        ents = []
        for para in essay.paragraphs:
            ents.extend([ent_l for ent_list in para.entities for ent_l in ent_list])

        entities.append(ents)
    if train:
        vectorizer = CountVectorizer(analyzer=identity)
        vectorizer = vectorizer.fit(entities)
        pickle.dump(vectorizer, open("../models/entity_vec.pkl", "wb"))
    else:
        vectorizer = pickle.load(open("../models/entity_vec.pkl", "rb"))
    X_entities = vectorizer.transform(entities)
    X_entities = pd.DataFrame(X_entities.todense(), columns=vectorizer.get_feature_names_out())
    return X_entities


def num_linguistic_errors(essays, train=True):
    """
    Feature for the number of errors, normalized number of errors and max number of errors per sentence


    :param essays: List of essays
    :param train: Boolean indicator if fitted vectorizer should be stored for later prediction
    :return: Pandas dataframe (examples x features)
    """

    features = []
    for essay in essays:
        max_errors = 0
        for para in essay.paragraphs:
            for offset in para.offsets:
                actual_errors = 0
                for match in essay.ling_errors:
                    if offset[0] <= match.offset < offset[1]:
                        actual_errors += 1
                    if offset[1] < match.offset:
                        break
                if actual_errors > max_errors:
                    max_errors = actual_errors

        # max errors
        features.append([len(essay.ling_errors), len(essay.ling_errors) / len(essay), max_errors])

    return pd.DataFrame(features, columns=['Linguistic Errors', "Error per Token", 'Max Errors per Sentence'])


def set_linguistic_errors(essays):
    """
    Check for linguistic errors based on LanguageTool (wrapper).
    Match is set as essay.ling_errors

    :param essays: List of essays
    :return:
    """
    tool = language_tool_python.LanguageTool('en-US')
    for index, essay in enumerate(essays):
        essay.ling_errors = tool.check(essay.text)
    return


def set_embeddings(essays):
    """
    Set paragraph embeddings obtained from sBERT.


    :param essays: List of essays
    :return:
    """
    texts = []
    for essay in essays:
        texts.append(essay.title)
        for para in essay.paragraphs:
            texts.append(" ".join(para.sentences))

    embeddings = embedding_encoding(texts)

    # with open('data/embeddings2.pkl', "wb") as fOut:
    #     pickle.dump(embeddings, fOut)

    index = 0
    for essay in essays:
        essay.title_embeddings = embeddings[index].cpu()

        index += 1
        for para in essay.paragraphs:
            para.paragraph_embeddings = embeddings[index].cpu()
            #print(para.paragraph_embeddings.is_cuda)
            index += 1


def metadiscourse_number(essays, train=True):
    """
    Occurences of metadiscourse words per category.


    :param essays: List of essays
    :param train:
    :return: Pandas dataframe (examples x features)
    """
    meta_markers = []
    for essay in essays:
        meta_markers_essay = []
        for k in metadiscourse:
            p = regex.compile(r"(\b\L<options>)\b", options=metadiscourse[k])
            count = len(regex.findall(p, essay.text.lower()))

            meta_markers_essay.append(count / len(essay))
        meta_markers.append(meta_markers_essay)
    return pd.DataFrame(meta_markers, columns=[k for k in metadiscourse])


def prompt_similary(essays, train=True):
    """
    Calculate maximum, minimum and mean similarity to prompt-

    :param essays: List of essays
    :param train:
    :return: Pandas dataframe (examples x features)
    """
    feature = []
    for essay in essays:
        prompt = essay.title_embeddings
        similarities = []
        for para in essay.paragraphs:
            similarities.append(util.cos_sim(prompt, para.paragraph_embeddings).item())
        feature.append([max(similarities), min(similarities), mean(similarities)])
    return pd.DataFrame(feature, columns=["Max Prompt Similarity", "Min Prompt Similarity", "Mean Prompt Similarity"])


def set_sentence_offsets(essays):
    """
    Set beginning and ending index of sentence boundaries.

    :param essays:
    :return:
    """
    for essay in essays:
        index = 0
        for para in essay.paragraphs:
            para.offsets = []
            for sentence in para.sentences:
                begin = essay.text.find(sentence, index)
                end = begin + len(sentence)
                index = end
                para.offsets.append((begin, end))


def set_sentiment(essays):
    """
    Set the sentiment for each sentence in each paragraph.


    :param essays: List of sentences
    :return:
    """
    sentiment_analysis = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    for essay in essays:
        for para in essay.paragraphs:
            para.sentiment = []
            sent = sentiment_analysis(para.sentences)
            for s in sent:
                label = s['label']
                if label == 'LABEL_0':
                    value = s['score'] * -1
                if label == 'LABEL_1':
                    value = s['score'] * 0
                if label == 'LABEL_2':
                    value = s['score']
                para.sentiment.append(value)


def adu(essays, train=True):
    """
    Calculate all adu related features (adu_composition_feature, adu_flow_featuresm, adu_ngram_features)


    :param essays: List of essays
    :param train:  Boolean indicator if fitted vectorizer should be stored for later prediction
    :return: Pandas dataframe (examples x features)
    """
    feature_func = [adu_composition_feature,
                    adu_flow_features, adu_ngram_features]
    features = []
    for func in feature_func:
        features.append(func(essays, train=train))
    return pd.concat(features, axis=1)


def content_style(essays, train=True):
    """
    Caluclate token and pos-ngram features.

    :param essays: List of essays
    :param train:  Boolean indicator if fitted vectorizer should be stored for later prediction
    :return: Pandas dataframe (examples x features)
    """
    feature_func = [pos_ngram_features, token_ngrams]
    features = []
    for func in feature_func:
        features.append(func(essays, train=train))
    return pd.concat(features, axis=1)


def similarity(essays, train=True):
    """
    Calculate prompt to paragraph similarity features.

    :param essays:  List of essays
    :param train:  Boolean indicator if fitted vectorizer should be stored for later prediction
    :return: Pandas dataframe (examples x features)
    """
    feature_func = [similarity_flow, prompt_similary]
    features = []
    for func in feature_func:
        features.append(func(essays, train=train))
    return pd.concat(features, axis=1)


def experiment(essays, train=True, type=''):
    """
    Prepare features for each of the 5-fold cross-validation splits.
    Features are used to find the best combination.
    Dumped at ../data/feature_names_<type>.pkl

    :param essays: List of essays
    :param train:  Boolean indicator if fitted vectorizer should be stored for later prediction
    :param type: String 'org', 'arg', 'th' to indicate quality dimension
    :return:
    """
    feature_func = [essay_todf, num_linguistic_errors, sentiment_flow, adu, content_style, length_stats, function_flow,
                    entities, similarity, metadiscourse_number]
    if type == 'org':
        essays = [essay for essay in essays if essay.annotation.organization_fold is not None]
    if type == 'arg':
        essays = [essay for essay in essays if essay.annotation.arg_strength_fold is not None]
    if type == 'th':
        essays = [essay for essay in essays if essay.annotation.thesis_clarity_fold is not None]

    features_dict = {}
    essay_fold_test = []
    essay_fold_train = []
    for i in range(0, 5):
        if type == 'org':
            essay_fold_test = [essay for essay in essays if essay.annotation.organization_fold == i]
            essay_fold_train = [essay for essay in essays if essay.annotation.organization_fold != i]
        if type == 'arg':
            essay_fold_test = [essay for essay in essays if essay.annotation.arg_strength_fold == i]
            essay_fold_train = [essay for essay in essays if essay.annotation.arg_strength_fold != i]
        if type == 'th':
            essay_fold_test = [essay for essay in essays if essay.annotation.thesis_clarity_fold == i]
            essay_fold_train = [essay for essay in essays if essay.annotation.thesis_clarity_fold != i]

        X_features = []
        for func in feature_func:
            print(func.__name__, str(i))
            t0 = time.time()
            feature_train = func(essay_fold_train, train=True)
            feature_test = func(essay_fold_test, train=False)
            X_feature_df = pd.concat([feature_train, feature_test], axis=0)
            X_features.append(X_feature_df)
            if func.__name__ in features_dict:
                features_dict[func.__name__].append(X_feature_df)
            else:
                features_dict[func.__name__] = [X_feature_df]

        X_features_df = pd.concat(X_features, axis=1)
        X_features_df.set_index('Essay', inplace=True)
        X_features_df.to_csv('../data/data_features\\' + type + str(i) + 'Features.csv', sep="\t")
    names_and_features = [(k, v) for k, v in features_dict.items()]
    pickle.dump(names_and_features, open('../data/feature_names_' + type + '.pkl', "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def transform(essays, train=True, type='all'):
    """


    :param essays: List of essays
    :param train:
    :param type:
    :return:
    """
    t0 = time.time()
    feature_func = [essay_todf, num_linguistic_errors, sentiment_flow, adu, content_style, length_stats, function_flow,
                    entities, similarity, metadiscourse_number]
    if type == 'strength':
        feature_func = [essay_todf, content_style, adu, function_flow, similarity]
    if type == 'organization':  # adu content_stylee
        feature_func = [essay_todf, num_linguistic_errors, adu, length_stats, function_flow, similarity,
                        metadiscourse_number]
    if type == 'clarity':
        feature_func = [essay_todf, num_linguistic_errors, sentiment_flow, content_style, length_stats, similarity]

    X_features = []
    for func in feature_func:  # t0 = time.time()
        X_features.append(func(essays, train=train))

    feature_names = [func.__name__ for func in feature_func]
    names_and_features = list(zip(feature_names, X_features))
    pickle.dump(names_and_features, open('../data/feature_names.pkl', "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    X_features_df = pd.concat(X_features, axis=1)
    X_features_df.set_index('Essay', inplace=True)
    return X_features_df


def transform_dimensions(essays, train=False, store=True):

    """
    Calculate the best feature combination for each dimension.

    :param essays: List of essays
    :param train:  Boolean indicator if fitted vectorizer should be stored for later prediction
    :param store:  Boolean indicator whether to store the features as csv at ../data/data_features\\Features_<type>.csv
    :return: Pandas dataframe of features for each quality dimension X_arg_df, X_org_df, X_th_df
    """


    feature_arg = [essay_todf, adu, content_style, similarity]
    feature_org = [essay_todf, num_linguistic_errors, content_style, length_stats, entities, similarity]
    feature_th = [essay_todf, num_linguistic_errors, sentiment_flow, function_flow, entities, similarity]

    all_features = feature_arg + feature_org + feature_th
    feature_func= list(dict.fromkeys(all_features).keys())



    feature_arg_names = [feature.__name__ for feature in feature_arg]
    feature_org_names = [feature.__name__ for feature in feature_org]
    feature_th_names = [feature.__name__ for feature in feature_th]

    X_features = []
    for func in feature_func:
        X_features.append(func(essays, train=train))

    feature_names = [func.__name__ for func in feature_func]
    names_and_features = list(zip(feature_names, X_features))

    arg_df = [df[1] for df in names_and_features if df[0] in feature_arg_names]
    org_df = [df[1] for df in names_and_features if df[0] in feature_org_names]
    th_df = [df[1] for df in names_and_features if df[0] in feature_th_names]



    X_arg_df = pd.concat(arg_df, axis=1)
    X_arg_df.set_index('Essay', inplace=True)
    X_org_df = pd.concat(org_df, axis=1)
    X_org_df.set_index('Essay', inplace=True)
    X_th_df = pd.concat(th_df, axis=1)
    X_th_df.set_index('Essay', inplace=True)
    if store:
        X_arg_df.to_csv('../data/data_features\\Features_arg.csv', sep="\t")
        X_org_df.to_csv('../data/data_features\\Features_org.csv', sep="\t")
        X_th_df.to_csv('../data/data_features\\Features_th.csv', sep="\t")

    return X_arg_df, X_org_df, X_th_df


def essay_todf(essays, train=False):
    df_essay = []
    for essay in essays:
        df_essay.append([essay.file])
    return pd.DataFrame(df_essay, columns=["Essay"])


def essay_to_df(essays):
    df_essay = []
    for essay in essays:
        df_essay.append([essay.file, essay.text])
    return pd.DataFrame(df_essay, columns=["Essay", "Text"])


def prepare_datasets(path="../data/ICLEv3/texts\\", title_path='../data/ICLEv3/metadata.xlsx', sheet_name='metadata',
                     store_path='../data/iclev3_essays.pkl', version='v3'):
    """

    Processes raw text essays to essay objects with annotations if given.
    Sets adu_flow and stores essay in pickle format.

    :param path: Location of raw essay text files
    :param title_path: Path to the ICLE metadata
    :param sheet_name: Sheetname of metadata file that contains the titles
    :param store_path: store annotated essays in pickle format
    :param version: 'v2' for ICLE v2 or 'v3' fpr ICLE corpus v3
    :return: essays
    """
    print("Get Annotations")
    annotations = process_annotations()
    print("Get Files")
    essays = get_essays(path=path, title_path=title_path, sheet_name=sheet_name, annotations=annotations,
                        version=version)
    print("Set ADUs")
    print(len(essays))
    set_adu(essays)
    pickle.dump(essays, open(store_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    json_essay = jsonpickle.encode(essays, indent=4)
    with open("../data/essays.json", "w") as outfile:
        outfile.write(json_essay)
    print("Essays stored")
    return essays




def essay_quality(essays=None, prepare=False, store_path='../data/essays_ann.pkl', features=True):
    """
    Set all metainformation for later feature computation.


    :param essays: List of essays
    :param prepare: Whether to process essays if not already done
    :param store_path: Path to store annoated essays
    :param features:
    :return:
    """
    if prepare:
        prepare_datasets()
    if essays is None:
        essays = pickle.load(open("../data/essays.pkl", 'rb'))

    print(len(essays))
    print('Linguistic Errors')
    set_linguistic_errors(essays)
    print('Flows')
    set_flow_variations(essays)
    set_sentence_labels(essays)
    set_para_flow(essays)
    print('Entities')
    set_entities(essays)
    pickle.dump(essays, open(store_path, "wb"))
    print('Sentiment')
    set_sentiment(essays)
    set_sentence_offsets(essays)
    print('Embeddings')
    set_embeddings(essays)
    pickle.dump(essays, open(store_path, "wb"))

    print("Get Features")

    # if features:
    #     X_features = transform(essays)
    #     X_features.to_csv('../data/data_features\\Features.csv', sep="\t")


def features_from_essays(essays_path='data/essays_ann.pkl', type='all'):
    essays = pickle.load(open(essays_path, 'rb'))
    X_features = transform(essays, type=type)
    X_features.to_csv('data_features\\Features.csv', sep="\t")


def essay_text_df():
    essays = pickle.load(open("data/essays_ann.pkl", 'rb'))
    essays_text = essay_to_df(essays=essays)
    essays_text.to_csv('data\\essays_text.csv', sep="\t")

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Feature generation and quality evaluation')

    parser.add_argument('-experiments', action='store_true',
                        help='Creates essay objects to replicate the experiments by Persing et. al')

    parser.add_argument('-annotate_iclev3', action='store_true',
                        help='Creates essay objects with all the data needed for the corpus creation')

    args = parser.parse_args()
    if args.experiments:
        prepare_datasets(path="../data/data_essays\\", title_path='../data/ICLEv2/GERM_corrected_ICLE_metadata.xls',
                         sheet_name='GERM_corrected', store_path='../data/essays.pkl', version='v2')
        essays = pickle.load(open("../data/essays.pkl", 'rb'))
        essay_quality(essays, store_path='../data/essays_ann.pkl')
        print("Export features for experiments")
        experiment(essays=essays, type='org')
        experiment(essays=essays, type='th')
        experiment(essays=essays, type='arg')
        print("Export features for deployment")
        transform_dimensions(essays=essays, train=True)

    if args.annotate_iclev3:
        prepare_datasets(path="../data/ICLEv3/texts\\", title_path='../data/ICLEv3/metadata.xlsx', sheet_name='metadata', store_path='../data/iclev3_essays.pkl', version='v3')
        essays = pickle.load(open("../data/iclev3_essays.pkl", 'rb'))
        essay_quality(essays, store_path='../data/iclev3_essays_ann.pkl')

        essays = pickle.load(open("../data/iclev3_essays.pkl", 'rb'))




