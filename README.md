# Personal_Chatbot
The personal chatbot can be used to answer from the data you have. you can import pdfs then it will answer on the go. there are several modifications we can to do fine-tune LLM to get the end answer perfectly. such that we can say that this a versatile chatbot. 

## Functionalities
- User-friendly web interface
- PDFs can be uploaded to use as reference files 
- Can remember the chat history and display on the page
- The data sent to the LLM can be seen.
- Chat history can be erasable which means you can start from the beginning whenever you want
  
## Requirements
- Python >=3.10
- OpenAI API key
  - You will need to fill the `OPENAI_API_KEY` of the `.env` file in the repository.
- Other requirements
  - Other requirements can be installed with the help of the requirements.txt file.
  - `pip install -r requirement.txt` or `pip3 install -r requirement.txt`

  
## Installation and Usage
1. Clone the repository.
  - `git clone https://github.com/vakeesank99/Personal-Chatbot.git`
2. Install the necessary dependencies.
  - `pip install -r requirement.txt` or `pip3 install -r requirement.txt`
3. Update the environment variable.
  - Open the .env file and update the `OPENAI_API_KEY` with your own OpenAI API key.
  - Follow the link to create your API key. [OpenAI API](https://platform.openai.com/api-keys)
4. Run the program.
  - `python panel_app.py` or `python3 panel_app.py`

## Further improvement
- Include Graph-based data storage and retrieval methods rather than vector spaces.
- Instead of panels, other platforms like Streamlit and Gradio will be used for UI design
- Include more importing formats such as .docx, audio and video formats.
