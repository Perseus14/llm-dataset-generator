import glob
import os
import random
import ast
import logging
import pandas as pd
import re

from llmdg.utils.chroma import ChromaDB
from llmdg.utils.llm import call_gpt
from llmdg.utils.prompts import SYSTEM_PROMPT, USER_PROMPT, PROMPT_THRESH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s",
    handlers=[logging.FileHandler("llm.log"), logging.StreamHandler()],
)

def process_docs(corpus_path, chunk_size=1000):
    logging.info("Creating Chunks of Text...")
    doc_list = glob.glob(os.path.join(corpus_path, '**', '*'), recursive=True)
    chromadb_instance = ChromaDB(doc_list, chunk_size)
    return chromadb_instance

def process_results(results):
    pattern = r'question: (.*?)\nanswer: (.*?)\n'
    matches = re.findall(pattern, results, re.DOTALL)
    res = []
    for match in matches:
        question = match[0].strip()
        answer = match[1].strip()
        res.append({'question': question, 'answer': answer})
    return res

def generate_dataset(db_instance, model='gpt-3.5-turbo', num_pairs=100):
    logging.info("Generating Datasets...")
    ids = list(map(str, list(range(db_instance.count()))))
    doc_results = db_instance.get(ids)
    
    count = num_pairs//PROMPT_THRESH
    rem = num_pairs%PROMPT_THRESH
    all_results = []
    for epoch in range(count):
        ind, doc = random.choice(list(enumerate(doc_results['documents'])))
        doc_id = doc_results['metadatas'][ind]['doc_id']
        chunk_id = doc_results['ids'][ind]
        
        system_prompt = SYSTEM_PROMPT.format(num=PROMPT_THRESH)
        user_prompt = USER_PROMPT.format(context=doc)
        
        logging.info("Calling GPT: " + str(epoch) + '/' + str(count))
        results = call_gpt(model, system_prompt, user_prompt)
        results = process_results(results)
        all_results.extend([(results, doc, doc_id)])
    
    if rem!=0:
        ind, doc = random.choice(list(enumerate(doc_results['documents'])))
        doc_id = doc_results['metadatas'][ind]['doc_id']
        chunk_id = doc_results['ids'][ind]
        
        system_prompt = SYSTEM_PROMPT.format(num=rem)
        user_prompt = USER_PROMPT.format(context=doc)
        logging.info("Calling GPT: " + str(epoch+1) + '/' + str(count))
        results = call_gpt(model, system_prompt, user_prompt)
        results = process_results(results)
        all_results.extend([(results, doc, doc_id)])
        
    return all_results

def process_dataset(dataset):
    processed_dataset = []
    
    for json_list, doc, doc_id in dataset:
        for ele in json_list:
            ele['doc_id'] = doc_id
            ele['context'] = doc
        processed_dataset.extend(json_list)
        
    dataset_df = pd.DataFrame(processed_dataset)
    return dataset_df
            
        

def run(corpus_path, output_path=None, model='gpt3.5-turbo', num_pairs=100, chunk_size=1000):
    logging.info("Starting Process...")
    db = process_docs(corpus_path=corpus_path, chunk_size=chunk_size)
    dataset = generate_dataset(db_instance=db, model=model, num_pairs=num_pairs)
    try:
        df_results = process_dataset(dataset=dataset)
        df_results = df_results.rename_axis('id').reset_index()
        df_results = df_results[['id','doc_id', 'context', 'question', 'answer']]
        if output_path is not None:
            df_results.to_csv(output_path, index=False)
        return df_results
    except Exception as e:
        logging.error("Error: " + str(e))
        return dataset
        
    
if __name__=='__main__':
    corpus_path = '../examples/UnitedAirlines'
    results = run(corpus_path)
    