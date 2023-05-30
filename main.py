from llmdg import generate
from llmdg import evaluate
from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.getenv("API_KEY")

if __name__=='__main__':
    corpus_path = 'examples/corpus_folder'
    output_path = 'datasets/corpus.csv'
    
    results_gen = generate.generate(corpus_path, output_path, model='gpt-3.5-turbo', num_pairs=100, chunk_size=1000)
    
    llm_out_path = 'evaluate/llm_output.csv'
    ground_truth_out_path = 'datasets/corpus.csv'
    result_path = 'results/results_example.csv'
    
    results_eval = evaluate.evaluate(llm_out_path, ground_truth_out_path, result_path)