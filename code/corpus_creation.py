import time
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import lightgbm
import glob
from sklearn.preprocessing import minmax_scale
from scipy.special import softmax
from main import set_flow_variations, set_para_flow, transform_dimensions, identity, find_ngrams
from predict import embedding_encoding
from utils import store_essays_json
import spacy
import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np
from multiprocessing import Pool
import glob
import os
import torch
import warnings
import time

from transformers import BertTokenizer, BertForNextSentencePrediction
import copy
from sentence_transformers import SentenceTransformer, util


import nltk.data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

nlp = spacy.load('en_core_web_sm')



def remove_sent(essay, indices):
    '''
    Removes all associate information from the sentencence in paragraph indicies[0] and sentence indices[1]

    :param essay: Annotated essay object
    :param indices: Tuple with paragraph and sentence index
    :return:
    '''
    sent = essay.paragraphs[indices[0]].sentences.pop(indices[1])
    essay.paragraphs[indices[0]].sentence_labels.pop(indices[1])
    essay.paragraphs[indices[0]].entities.pop(indices[1])
    essay.paragraphs[indices[0]].adu_flow.pop(indices[1])
    essay.paragraphs[indices[0]].sentiment.pop(indices[1])
    essay.paragraphs[indices[0]].pos_tags.pop(indices[1])
    essay.paragraphs[indices[0]].tokens.pop(indices[1])
    essay.paragraphs[indices[0]].paragraph_embeddings = None
    offset = essay.text.find(sent)
    length = len(sent)
    for match in essay.ling_errors:
        if offset <= match.offset <= (offset + length):
            essay.ling_errors.remove(match)
    essay.text = essay.text.replace(sent, '')
    set_para_flow([essay])
    set_flow_variations([essay])
    essay.paragraphs[indices[0]].paragraph_embeddings = embedding_encoding(
        (" ".join(essay.paragraphs[indices[0]].sentences))).cpu()


def define_candidates(essays, checkpoint=True):
    '''
    Check if the sentence could be removed without resulting in a random sequence of sentences.
    Runs BERT with NSP head

    :param essays: List of essays
    :param checkpoint: Boolean value if intermediate results should be stored
    :return:
    '''
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    for iteration, essay in enumerate(essays):
        # print(essay.file)
        sentences = [essay.title]
        for para in essay.paragraphs:
            sentences.extend(para.sentences)
        sentences.extend(['', ''])
        candidates = []
        for i in range(0, len(sentences) - 3):
            j = i + 2
            encoding = tokenizer(sentences[i], sentences[j], return_tensors="pt", max_length=512, truncation=True)
            outputs = model(**encoding)
            candidates.append(torch.argmax(outputs.logits).item())
        index = 0
        for idx, para in enumerate(essay.paragraphs):
            para.candidates = candidates[index:index + len(para.sentences)]
            index = index + len(para.sentences)
        if checkpoint:
            store_essays_json([essay], path="../data/corpus/candidates/")


def score_essays(essays):
    '''
    Sets essay score for each essay with the 1/3*(thesis clarity + argument strength + organization)

    :param essays: List of annotated essays
    :return:
    '''
    w_org = 1 / 3
    w_arg = 1 / 3
    w_th = 1 / 3
    model_org = lightgbm.Booster(model_file='../models/organization.txt')
    model_th = lightgbm.Booster(model_file='../models/thesis_clarity.txt')
    model_arg = lightgbm.Booster(model_file='../models/arg_strength.txt')
    X_arg, X_org, X_th = transform_dimensions(essays, train=False)
    org_scores = model_org.predict(X_org.iloc[:, 0:].to_numpy())
    arg_scores = model_arg.predict(X_arg.iloc[:, 0:].to_numpy())
    th_scores = model_th.predict(X_th.iloc[:, 0:].to_numpy())
    for idx, essay in enumerate(essays):
        # print(essay.file, arg_scores[idx])
        essay.score = org_scores[idx] * w_org + arg_scores[idx] * w_arg + th_scores[idx] * w_th
        # print("Arg: ", arg_scores[idx],"Org: ", org_scores[idx], "Th: ",th_scores[idx], essay.score)


def candidate_sent_indices(essays):
    """
    Define which sentences should be considered.
    Considers all sentences which are an ADU and can be removed without resulting in a random sentence sequence.

    :param essays: List of annoated essays
    :return: Dictionary with essay file name as key and list of indices of sentence positions
    """
    essays_dict = {}
    for idx, essay in enumerate(essays):
        for i, para in enumerate(essay.paragraphs):
            para.sent_scores = [0] * len(para.sentences)
            if len(para.sentences) > 3:
                for j, sent in enumerate(para.sentences):
                    # print(para.adu_flow, para.candidates, idx)
                    if para.candidates[j] == 0 and para.adu_flow[j] != 'None':
                        if essay.file in essays_dict:
                            essays_dict[essay.file].append((i, j))
                        else:
                            essays_dict[essay.file] = [(i, j)]
    return essays_dict


def set_sent_quality(essays, checkpoint_path='../data/corpus/icle_corpus_ranked/', checkpoint_steps=10):
    """

    :param essays: List of annotated essays
    :param checkpoint_path: Path to store essays as json. Essays sentence quality is estimated. Defaults to ../data/corpus/icle_corpus_ranked/'
    :param checkpoint_steps: Number of files that should be processed before being written to the file system
    :return:
    """
    essays_dict = candidate_sent_indices(essays)
    essays_to_store = []
    for iteration, essay in enumerate(essays):
        essay_var = []
        if essay.file in essays_dict:
            index = essays_dict[essay.file]
            for indices in index:
                essay_est = copy.deepcopy(essay)
                remove_sent(essay_est, indices)
                essay_var.append(essay_est)
            score_essays(essay_var)
            for idx, e in enumerate(essay_var):
                indices = index[idx]
                essay.paragraphs[indices[0]].sent_scores[indices[1]] = (essay.score - e.score)

        essays_to_store.append(essay)

        if iteration % checkpoint_steps == 0 and checkpoint_path is not None:
            store_essays_json(essays_to_store, path=checkpoint_path)
            essays_to_store = []

    store_essays_json(essays, path=checkpoint_path)


def create_quality_annotations(essays, score=True, candidates=True, quality=True):
    '''
    Create json that represent the processed essays. Essays are scored, candidates indicator is set and the argument components' qualities are estimated.

    :param essays: List of annotated essays
    :param score: Boolean indicator whether essays need to be scored
    :param candidates: Boolean indicator whether candidate indicator needs to be set
    :param quality: Boolean indicator whether sentence quality needs to be estimated
    :return: List of essays
    '''
    if score:
        score_essays(essays)
        store_essays_json(essays, path="../data/corpus/scored_essays/", overwrite=True)
    if candidates:
        define_candidates(essays)
        store_essays_json(essays, path="../data/corpus/candidates/", overwrite=True)
    if quality:
        set_sent_quality(essays)
        store_essays_json(essays, path="../data/corpus/icle_corpus_ranked/")

    return essays


def parallel_set_sent_quality(essays):
    essays_chunks = list(np.array_split(essays, 5))
    with Pool(5) as p:
        essays_chunks = p.map(set_sent_quality, essays_chunks)
    essays = list(np.asarray(essays_chunks).flatten())


def annotate_corpus_parallel(essays=None, essay_path='data/essays_ann.pkl'):
    '''
    Annotate essays with candidate indicator, score and sentence quality estimates.
    Runs the corpus annotation in parallel with poolsize 5.

    :param essays: List of essays
    :param essay_path: Path to load essays when essay are set to None
    :return:
    '''
    if essays is None:
        essays = pickle.load(open(essay_path, 'rb'))

    essays_chunks = list(np.array_split(essays, 5))
    with Pool(5) as p:
        essays_chunks = p.map(create_quality_annotations, essays_chunks)

    essays = list(np.asarray(essays_chunks).flatten())


def annotate_corpus(essays=None):
    if essays is None:
        essays = pickle.load(open('data/essays_ann.pkl', 'rb'))
    t0 = time.time()
    create_quality_annotations(essays)
    t1 = time.time()
    print(t1 - t0)


def power_method(M, epsilon=0.00001, max_iter=10000):
    '''
    Power iteration for eigenvector computation.

    :param M: 2D numpy array
    :param epsilon: error threshold defaults to 0.00001
    :param max_iter: maximum number of iterations
    :return: stationary distribution
    '''
    t = 0
    p_t = (np.ones((M.shape[0], 1)) * (1 / M.shape[0]))
    for i in range(max_iter):
        t = t + 1
        p_prev = p_t
        p_t = np.dot(M.T, p_t)
        residual = np.linalg.norm(p_t - p_prev)

        if residual < epsilon:
            break
    return p_t


def normalize(m):
    '''
    Normalize 2D numpy array on axis 1 using softmax.

    :param m: 2D numpy array
    :return:
    '''
    return softmax(m, axis=1)


def quality_surfer(essays):
    '''
    Create quality matrix with m_ij representing the quality of sentence j
    :param essays: List of annotated essays (sentence score estimates have to be set)
    :return: List of quality matrices
    '''
    matrices = []
    for essay in essays:
        matrix = []
        avg = 0
        num_scored = 0
        for para in essay.paragraphs:
            if len(para.sentences) > 3:
                for index, score in enumerate(para.sent_scores):
                    if para.candidates[index] == 0 and para.adu_flow[index] != 'None':
                        avg += score
                        num_scored += 1
        if avg != 0:
            avg = avg / num_scored
        quality_estimates = [avg]
        for para in essay.paragraphs:
            for index, score in enumerate(para.sent_scores):
                if len(para.sentences) > 3 and para.candidates[index] == 0 and para.adu_flow[index] != 'None':
                    quality_estimates.append(score)

                else:
                    quality_estimates.append(avg)
        matrix.append(minmax_scale(quality_estimates))

        matrix = np.repeat(matrix, len(matrix[0]), axis=0)
        matrices.append(normalize(matrix))
    return matrices


def centrality_score(essays):
    '''
    Compute cosine-similaritsy matrix based on sBERT embedding for each essay.

    :param essays: List of annotated essays
    :return: List of similarity matrices
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    essays_sentences = []
    for essay in essays:
        sentences = [essay.title]
        for para in essay.paragraphs:
            sentences.extend(para.sentences)
        embeddings.append(model.encode(sentences, convert_to_tensor=True).cpu())
        essays_sentences.append(sentences)
    sim_matrices = []
    for embedding in embeddings:
        sim_matrices.append(normalize(np.asarray((util.cos_sim(embedding, embedding)))))
    return sim_matrices


def rank_essay_sentences(essays, d=0.5, store_path='../data/corpus/icle_corpus_ranked/', checkpoint_steps=100):
    '''
    Compute PageRank for argument components based on quality and centrality

    :param essays: List of annotated essays (quality estimates have to be set)
    :param d: (damping factor) weight put to the quality matrix with (1-d) put to the centrality matrix
    :param store_path: Path to stored the final corpus with ranked essay. Defaults to ../data/corpus/icle_corpus_ranked/
    :param checkpoint_steps: Number of files that should be processed before being written to the file system
    :return:
    '''
    cent_scores = centrality_score(essays)
    quality_matrices = quality_surfer(essays)
    annotated_essays = []
    for index, essay in enumerate(essays):
        matrix_rank = cent_scores[index] * (1 - d) + d * quality_matrices[index]
        matrix_rank = power_method(matrix_rank, epsilon=0.0001)
        index = 1
        for idx, para in enumerate(essay.paragraphs):
            para.sent_ranks = matrix_rank[index:index + len(para.sentences)].flatten()
            index = index + len(para.sentences)
            annotated_essays.append(essay)
        if index % checkpoint_steps == 0:
            store_essays_json(essays=annotated_essays, path=store_path, overwrite=True)
            annotated_essays = []
    store_essays_json(essays=annotated_essays, path=store_path)


def candidates_from_checkpoint(folder_name='icle_corpus_ranked'):
    '''


    :return:
    '''
    essays = pickle.load(open('data/iclev3_essays_ann.pkl', 'rb'))
    annotated_essay_names = []
    for filename in glob.iglob('../data/corpus/' +folder_name+ '/*.json', recursive=True):
        annotated_essay_names.append(os.path.split(filename)[-1].split(".")[0])
    print(annotated_essay_names)
    not_annotated_essays = []
    for essay in essays:
        if essay.file not in annotated_essay_names:
            not_annotated_essays.append(essay)
    annotate_corpus(not_annotated_essays)


if __name__ == "__main__":
    essays = pickle.load(open("../data/iclev3_essays_ann.pkl", 'rb'))
    annotate_corpus_parallel(essays)
    rank_essay_sentences(essays)
