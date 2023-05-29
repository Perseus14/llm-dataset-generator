# LLM Dataset Generator

## Description

This project is designed to generate dataset from a corpus for training/evaluating LLMs. It utilises OpenAI's GPT models.


## Installation

Follow the instructions below to get the project up and running on your local machine:

1. **Clone** the repository: `git clone https://github.com/Perseus14/llm-dataset-generator.git`
2. **Navigate** to the project directory: `cd llm-dataset-generator`
3. **Create** virtual environment: `virtualenv -p /usr/bin/python3 venv`
4. **Activate** virtual environment: `source venv/bin/activate`
5. **Install** dependencies: `pip install -r requirements.txt`

## Usage

To use this project, follow these steps:

1. Create a folder and and the corpus to that folder (Supports txt and pdf files)
2. Create a .env file and add openAI apikey (Similar to .env_local)
3. Modify main.py and the required paths

## Pending Tasks

1. Generate conversational dataset
2. Add other LLM models
3. Modify to provide more control to users

## Contributing

Contributions are welcome! Follow the steps below to contribute to this project:

1. Fork the repository
2. Create a new branch: `git checkout -b new-feature`
3. Make your changes and commit them: `git commit -m 'Add new feature'`
4. Push the changes to your forked repository: `git push origin new-feature`
5. Open a pull request on the original repository

Please ensure that your contributions align with the project's coding style and guidelines.

## License

Apache 2.0
