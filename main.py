import chainlit as cl 
import openai 
import os 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.utilities import SearchApiAPIWrapper



os.environ['OPENAI_API_KEY'] = "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"
os.environ["SEARCHAPI_API_KEY"] = "Z5nPhqhpdGLHwPMnheZywYdC"

openai.api_key="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"

welcome_message = """Hello, I am Moneai, a super-intelligent real estate broker certified by the Dubai Land Department and developed by Tech Haus Medy (THM). I will assist you in discovering the optimal property for investment by offering in-depth market analysis followed by managing the entire transaction process."""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

def process_pdf():
    loader = PyPDFLoader("Regulations.pdf")
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs 


def get_docsearch():
    docs = process_pdf()
    # Save data in the user session
    #cl.user_session.set("docs", docs)
    # Create a unique namespace for the file

    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch

# docsearch = get_docsearch()
    
# pdf_chain = RetrievalQA.from_chain_type(
#     ChatOpenAI(temperature=0, streaming=True),
#     chain_type="stuff",
#     verbose=True,
#     retriever=docsearch.as_retriever(max_tokens_limit=4097),
# )

# csv_chain = create_csv_agent(
#         ChatOpenAI(temperature=0, model="gpt-4"),
#         "transactions-2023-10-21.csv",
#         verbose=True,
#         agent_type=AgentType.OPENAI_FUNCTIONS,
#     )

# search = SearchApiAPIWrapper()


# #agent.run("Plot me the evolution of yearly transactions value")
# tools = [
    
#     Tool(
#         name="regulations QA System",
#         func=pdf_chain.run,
#         description="The user will ask you questions about regulations in dubai, a comprehensive guide on UAE residence visas for investors, entrepreneurs, freelancers, and employees of Emirati companies. and questions that covers various pathways to obtain these visas, including through property purchase, company establishment, and employment.",
#     ),
#     Tool(
#         name="CSV agent",
#         func=csv_chain.run,
#         description="If you're curious about the real estate market in Dubai, performance metrics, or averages, ask away, and we'll delve into our CSV data for insights. you can also give results about the queries by executing code. The dataset provides comprehensive information on property transactions, including details like transaction number, date, type, registration, property area, size, usage (e.g., residential), location details (nearest metro, mall, and landmark), and specifics about the property itself (number of rooms, parking, etc.). The data appears to be centered around property sales in various regions of the UAE.",
#     ),

#     Tool(
#         name="Intermediate Answer",
#         func=search.run,
#         description=
# """
# You will be asked to propose real estate options to the user,  you will fetch the net especially this url : https://www.luxhabitat.ae/apartments-for-sale/dubai/palm-jumeirah/viceroy-residences/penthouse-11686/ and give an answer like this one : 

# example : My budget is 20 million what are my options?

# answer : you have access to a range of upscale and lavish real estate options. These might include high-end apartments in prestigious neighborhoods like Downtown Dubai, Palm Jumeirah, or Emirates Hills, as well as luxurious villas and penthouses with top-tier amenities and stunning views. check this

# https://www.luxhabitat.ae/apartments-for-sale/dubai/palm-jumeirah/viceroy-residences/penthouse-11686/
# """
#     )

# ]

# template = """
    
#     you are Moneai a super-intelligent real estate broker certified by the Dubai Land Department and developed by Tech Haus Medy (THM). you will assist the user in discovering the optimal property for investment by offering in-depth market analysis followed by managing the entire transaction process. 

#     You have two Tools,  one for CSV processing, and the other for PDF parsing. 
#     If the question cannot be answered using neither the CSV tool  nor the PDF tool,  answer with "I don't know”
# """

# #prompt = PromptTemplate(template=template)

# llm = OpenAI(temperature=0)
# # No async implementation in the Pinecone client, fallback to sync
# #llm_chain = LLMChain(prompt=prompt, llm=llm)

# agent = initialize_agent(
#     tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )


# agent.run(
#     " My budget is 20 million what are my option ?  " 
# )

@cl.on_chat_start
async def start():
    # Sending an image with the local file path
    await cl.Message(content=welcome_message).send()

    docsearch = get_docsearch()
        
    pdf_chain = RetrievalQA.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        verbose=True,
        retriever=docsearch.as_retriever(max_tokens_limit=4097),
    )

    csv_chain = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-4"),
            "transactions-2023-10-21.csv",
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

    search = SearchApiAPIWrapper()


    #agent.run("Plot me the evolution of yearly transactions value")
    tools = [
        
        Tool(
            name="regulations QA System",
            func=pdf_chain.run,
            description="The user will ask you questions about regulations in dubai, a comprehensive guide on UAE residence visas for investors, entrepreneurs, freelancers, and employees of Emirati companies. and questions that covers various pathways to obtain these visas, including through property purchase, company establishment, and employment.",
        ),
        Tool(
            name="CSV agent",
            func=csv_chain.run,
            description="If you're curious about the real estate market in Dubai, performance metrics, or averages, ask away, and we'll delve into our CSV data for insights. you can also give results about the queries by executing code. The dataset provides comprehensive information on property transactions, including details like transaction number, date, type, registration, property area, size, usage (e.g., residential), location details (nearest metro, mall, and landmark), and specifics about the property itself (number of rooms, parking, etc.). The data appears to be centered around property sales in various regions of the UAE.",
        ),

        Tool(
            name="Intermediate Answer",
            func=search.run,
            description=
    """
    You will be asked to propose real estate options to the user,  you will fetch the net especially this url : https://www.luxhabitat.ae/apartments-for-sale/dubai/palm-jumeirah/viceroy-residences/penthouse-11686/ and give an answer like this one : 

    example : My budget is 20 million what are my options?

    answer : you have access to a range of upscale and lavish real estate options. These might include high-end apartments in prestigious neighborhoods like Downtown Dubai, Palm Jumeirah, or Emirates Hills, as well as luxurious villas and penthouses with top-tier amenities and stunning views. check this

    https://www.luxhabitat.ae/apartments-for-sale/dubai/palm-jumeirah/viceroy-residences/penthouse-11686/
    """
        )

    ]

    template = """
        
        you are Moneai a super-intelligent real estate broker certified by the Dubai Land Department and developed by Tech Haus Medy (THM). you will assist the user in discovering the optimal property for investment by offering in-depth market analysis followed by managing the entire transaction process. 

        You have two Tools,  one for CSV processing, and the other for PDF parsing. 
        If the question cannot be answered using neither the CSV tool  nor the PDF tool,  answer with "I don't know”
    """

    #prompt = PromptTemplate(template=template)

    llm = OpenAI(temperature=0)
    # No async implementation in the Pinecone client, fallback to sync
    #llm_chain = LLMChain(prompt=prompt, llm=llm)

    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )


    cl.user_session.set("chain", agent)



@cl.on_message
async def main(message:str):
    agent = cl.user_session.get("chain")

    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    answer = await cl.make_async(agent.run)(message, callbacks=[cb])
    await cl.Message(answer).send()



    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "assistant" , "content" : "You are Moneai, a super-intelligent real estate broker certified by the Dubai Land Department and developed by Tech Haus Medy (THM).You assist the user in discovering the optimal property for investment by offering in-depth market analysis followed by managing the entire transaction process."},
    #         {"role" : "user" , "content": message}
    #     ],
    #     temperature=1
    # ) 
    # await cl.Message(content=response['choices'][0]['message']['content']).send() 

