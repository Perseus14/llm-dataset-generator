from llmdg import generate
from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.getenv("API_KEY")

if __name__=='__main__':
    corpus_path = 'examples/corpus_folder'
    output_path = 'datasets/corpus.csv'
    results = generate.run(corpus_path, output_path, model='gpt-3.5-turbo', num_pairs=100, chunk_size=1000)