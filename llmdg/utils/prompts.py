

SYSTEM_PROMPT = '''Based on the given context, generate {num} question-answer pairs. 
The generated data will be used to finetune an LLM. Do not generate any other text. 
Please generate it in the following format

Format:

question: <question>\nanswer: <answer>\n\nquestion: <question>\nanswer: <answer>

Sample Output:

question: Hi\nanswer: Hello, How can I help you?\n\nquestion: What is your name?\nanswer: I am a customer service bot, my name is BoltChat.

'''

USER_PROMPT = '''Context: {context} '''

PROMPT_THRESH = 10