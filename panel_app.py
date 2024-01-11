import os
import openai
import sys
import panel as pn
import param
import numpy as np
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import panel as pn  # GUI
from dotenv import load_dotenv, find_dotenv 
# import datetime
from bokeh.sampledata.autompg import autompg
from sentence_transformers import CrossEncoder

persist_directory = 'docs/chroma/'
llm_name = "gpt-3.5-turbo"
chain_type = "stuff" #refine map_rerank stuff map_reduce
search_type= "mmr" #mmr similarity
no_db_results = 4
frame_width=1700
frame_height=600
sys.path.append('../..')
pn.extension()

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.environ['OPENAI_API_KEY']

# current_date = datetime.datetime.now().date()
# if current_date < datetime.date(2023, 9, 2):
#     llm_name = "gpt-3.5-turbo-0301"
# else:
    # llm_name = "gpt-3.5-turbo"

def augment_multiple_query(query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert engineering academics. "
            "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions."
        },
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content


def load_db(file, chain_type, k):
    embeddings = OpenAIEmbeddings()

    if (file==""):
        chroma_db = Chroma(embedding_function=embeddings,persist_directory=persist_directory)
        normal_retriever = chroma_db.as_retriever(search_type=search_type, search_kwargs={"k": k})

        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=llm_name, temperature=0), 
            chain_type=chain_type, 
            retriever=normal_retriever, 
            return_source_documents=True,
            return_generated_question=True,
        )
        return qa 

    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    # --------------create vector database from data
    chroma_db = Chroma.from_documents( documents=docs, 
                               embedding=embeddings, 
                               persist_directory=persist_directory
                               )
    
    # docarray_db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    # ----------------define retriever
    #docarray_retriever = docarray_db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    #normal retriver 
    # retriever = db.as_retriever()
    normal_retriever = chroma_db.as_retriever(search_type=search_type, search_kwargs={"k": k})
    #compression retriever 
    llm=ChatOpenAI(model_name=llm_name, temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
                                                            base_compressor=compressor,
                                                            base_retriever= normal_retriever
                                                            )  
    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    if (chain_type=="map_reduce"):
        qa = RetrievalQA.from_chain_type(
                                            model_name=llm_name,
                                            retriever=compression_retriever,
                                            chain_type=chain_type
                                        )
    elif (chain_type=="refine"):
        qa=RetrievalQA.from_chain_type(
                                        model_name=llm_name,
                                        retriever=compression_retriever,
                                        chain_type=chain_type
                                    )
    elif (chain_type=="stuff"):
        qa = ConversationalRetrievalChain.from_llm(
                                                    llm=ChatOpenAI(model_name=llm_name, temperature=0), 
                                                    chain_type=chain_type, 
                                                    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
                                                    retriever=compression_retriever, 
                                                    return_source_documents=True,
                                                    return_generated_question=True,
                                                )
    return qa 



class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    def __init__(self,  **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.loaded_file = ""
        self.qa = load_db(self.loaded_file,chain_type, no_db_results)
    
    def call_load_db(self, count):
        if count == 0 or file_input.value is None:  # init or no file specified :
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # local copy
            self.loaded_file = file_input.filename
            button_load.button_style="outline"
            self.qa = load_db("temp.pdf", chain_type, no_db_results)
            button_load.button_style="solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=frame_width)), scroll=True)
        
        # augmented_queries = augment_multiple_query(query)
        # tot_query = [query] + augmented_queries
        # chroma_db = Chroma(persist_directory=persist_directory)
        # chroma_results =  chroma_db.query(query_texts=tot_query, n_results=10, include=['documents', 'embeddings'])
        # retrieved_documents = chroma_results['documents']

        # # Deduplicate the retrieved documents
        # unique_documents = set()
        # for documents in retrieved_documents:
        #     for document in documents:
        #         unique_documents.add(document)

        # unique_documents = list(unique_documents)
        # pairs = []
        # for doc in unique_documents:
        #     pairs.append([query, doc])
        # cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        # scores = cross_encoder.predict(pairs)
        # que_id=np.max(scores)
        # pairs[que_id][0]

        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=frame_width)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=frame_width, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  #clears loading indicator when cleared
        return pn.WidgetBox(*self.panels,scroll=True)

    @param.depends('db_query ', )
    def get_lquest(self):
        if not self.db_query :
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query )
        )

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return 
        rlist=[pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=frame_width, scroll=True)

    @param.depends('convchain', 'clr_history') 
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=frame_width, scroll=True)
        rlist=[pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=frame_width, scroll=True)

    def clr_history(self,count=0):
        self.chat_history = []
        return 

cb = cbfs()

# pn.config.theme = 'dark'
file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput( placeholder='Enter text here…')

bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp) 

# jpg_pane = pn.pane.Image( './img/convchain.jpg')

tab1 = pn.Column(
    pn.panel(conversation,  loading_indicator=True, height=frame_height),
    pn.layout.Divider(),
     pn.Row(inp),
     pn.layout.Divider()
)
tab2= pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources ),
)
tab3= pn.Column(
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
)
tab4=pn.Column(
    pn.Row( file_input, button_load, bound_button_load),
    pn.Row( button_clearhistory, pn.pane.Markdown("" )),
    pn.layout.Divider()
)
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# --Intelligent Knowledge Management System--')),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3),('Configure', tab4))
).servable()

