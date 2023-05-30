import os
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def check_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity_score


def eval_question(llm_questions, gt_questions):
    res = []
    for llm_question, gt_question in zip(llm_questions, gt_questions):
        if llm_question == gt_question:
            res.append(1)
        else:
            res.append(0)
    return res

def eval_answer(llm_answers, gt_answers):
    res = []
    for llm_answer, gt_answer in zip(llm_answers, gt_answers):
        res.append(check_similarity(llm_answer, gt_answer))
    return res


def eval_context(llm_contexts, gt_contexts):
    res = []
    for llm_context, gt_context in zip(llm_contexts, gt_contexts):
        res.append(check_similarity(llm_context, gt_context))
    return res
    

def evaluate(llm_out_path, ground_truth_out_path, result_path):
    df_gt = pd.read_csv(ground_truth_out_path)
    df_llm = pd.read_csv(llm_out_path)
    
    gt_json = json.loads(df_gt.to_json(orient='columns'))
    llm_json = json.loads(df_llm.to_json(orient='columns'))
    
    res_context = eval_context(llm_json['context'].values(), gt_json['context'].values())
    res_question = eval_question(llm_json['question'].values(), gt_json['question'].values())
    res_answer = eval_answer(llm_json['answer'].values(), gt_json['answer'].values())
    
    res_df = pd.DataFrame()
    
    for feat in ['context', 'question', 'answer']:
        res_df['llm_' + feat] = llm_json[feat].values()
        res_df['gt_' + feat] = gt_json[feat].values()
    
    res_df['res_context'] = res_context
    res_df['res_question'] = res_question
    res_df['res_answer'] = res_answer
    
    res_df = res_df.rename_axis('id').reset_index()
    
    res_df.to_csv(result_path, index=False)
    return res_df
