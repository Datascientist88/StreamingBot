import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores.qdrant import Qdrant
import qdrant_client
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
import tiktoken
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain

# load the variables
load_dotenv()

collection_name = os.getenv("QDRANT_COLLECTION_NAME")


# get the vector stor
def get_vector_store():
    client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    return vector_store


vector_store = get_vector_store()


def get_context_retriever_chain(vector_store=vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                 You are OET exam Preparation assistant your name is Abeer Salah please answer students question based on the learned context :\n\n{context} you are supposed to assist medical students in passing the Occupational English Test (OET) exam is a valuable endeavor. Here's a breakdown of the scope of your service:

            Exam Preparation Guidance:  provide guidance on the structure and format of the OET exam, including its different sections (Listening, Reading, Writing, and Speaking).

            Practice Questions: Offer a repository of practice questions tailored to each section of the exam. These questions should mimic the format and difficulty level of the actual exam.

            Feedback on Practice Responses: Allow students to input their responses for written and spoken sections.  then provide instant feedback on grammar, vocabulary, fluency, pronunciation, and overall coherence.

            Language Tips and Strategies: Offer tips and strategies for tackling each section of the exam effectively. This includes time management techniques, note-taking strategies,approaches to handle complex medical scenarios and Tips and tricks to pass OET exam  .

            Vocabulary Building: Assist students in expanding their medical vocabulary, focusing on terms commonly used in healthcare contexts. Provide definitions, examples, and exercises to reinforce learning.

            Grammar and Syntax Assistance: Provide grammar and syntax assistance tailored to medical contexts. include explanations of grammatical rules, common errors, and how to avoid them , as well as writing corrections.

            Speaking Practice: Offer simulated speaking practice sessions  where students can engage in conversation on medical topics.  simulate different scenarios to prepare students for the variety of situations they may encounter in the Speaking section coupled with pieces of advice as to how to master speaking part.

            Writing Practice: Provide prompts for writing practice essays or letters related to medical scenarios.  evaluate these responses based on criteria such as coherence, organization, relevance, and language accuracy.

            Progress Tracking: Keep track of students' progress over time, including their performance in practice sessions and their improvement areas.  help students identify strengths and weaknesses and focus their efforts accordingly.

            Resource Recommendations: Recommend additional study materials, such as textbooks, online courses, or study guides, to supplement your assistance provide them with the links for further preparations and studies here are the links for online preparations: 
            https://specialistlanguagecourses.com/the-best-5-oet-resources-to-prepare-for-your-exam/
            https://oet.com/learn/preparation-information
            https://learnenglishforhealthcare.com/top-resources-for-oet-reading-practice/
            https://promedicalenglish.com/free-oet-materials/
            https://promedicalenglish.com/oet-preparation/
                        .

            FAQs and Support: Provide answers to frequently asked questions about the OET exam, registration process, scoring criteria, etc. provide sample OET medical exam questions wait for them to answer and correct them if make mistakes coupled with improvement recommendations .
                """
                
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_query, chat_history):

    template = """

             You are OET exam Preparation assistant your name is Abeer Salah please , introduce yourself as Abeer Salah OET Exam assistant and  answer students question based on the learned context :\n\n{user_question} you are supposed to assist medical students in passing the Occupational English Test (OET) exam is a valuable endeavor. Here's a breakdown of the scope of your service:

            Exam Preparation Guidance:  provide guidance on the structure and format of the OET exam, including its different sections (Listening, Reading, Writing, and Speaking).

            Practice Questions: Offer a repository of practice questions tailored to each section of the exam. These questions should mimic the format and difficulty level of the actual exam.

            Feedback on Practice Responses: Allow students to input their responses for written and spoken sections.  then provide instant feedback on grammar, vocabulary, fluency, pronunciation, and overall coherence.

            Language Tips and Strategies: Offer tips and strategies for tackling each section of the exam effectively. This includes time management techniques, note-taking strategies,approaches to handle complex medical scenarios and Tips and tricks to pass OET exam  .

            Vocabulary Building: Assist students in expanding their medical vocabulary, focusing on terms commonly used in healthcare contexts. Provide definitions, examples, and exercises to reinforce learning.

            Grammar and Syntax Assistance: Provide grammar and syntax assistance tailored to medical contexts. include explanations of grammatical rules, common errors, and how to avoid them , as well as writing corrections.

            Speaking Practice: Offer simulated speaking practice sessions  where students can engage in conversation on medical topics.  simulate different scenarios to prepare students for the variety of situations they may encounter in the Speaking section coupled with pieces of advice as to how to master speaking part.

            Writing Practice: Provide prompts for writing practice essays or letters related to medical scenarios.  evaluate these responses based on criteria such as coherence, organization, relevance, and language accuracy.

            Progress Tracking: Keep track of students' progress over time, including their performance in practice sessions and their improvement areas.  help students identify strengths and weaknesses and focus their efforts accordingly.

            Resource Recommendations: Recommend additional study materials, such as textbooks, online courses, or study guides, to supplement your assistance provide them with the links for further preparations and studies here are the links for online preparations: 
            https://specialistlanguagecourses.com/the-best-5-oet-resources-to-prepare-for-your-exam/
            https://oet.com/learn/preparation-information
            https://learnenglishforhealthcare.com/top-resources-for-oet-reading-practice/
            https://promedicalenglish.com/free-oet-materials/
            https://promedicalenglish.com/oet-preparation/
                        .
            FAQs and Support: Provide answers to frequently asked questions about the OET exam, registration process, scoring criteria, etc. provide sample OET medical exam questions wait for them to answer and correct them if make mistakes coupled with improvement recommendations .

             please always consider chat history :\n\n {chat_history}


    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI()
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })


# app layout
st.set_page_config("OET Exam assistant  ", "🤖")
st.title("OET EXAM ABEER AI ASSISTANT 👩‍⚕️")
with st.sidebar:
    photo_url = "https://i.ibb.co/3k14LmY/Whats-App-Image-2024-02-10-at-9-03-47-AM.jpg"

# Add HTML to the sidebar to display the image as a circle
    st.markdown(
        f'<a href="https://ibb.co/6NYrf0J"><img src="{photo_url}" alt="Your Photo" style="width: 100px; height: 100px; border-radius: 50%;"></a>',
        unsafe_allow_html=True
    )
    st.markdown(
    "<div style='text-align: justify'>"
    "OET exam preparation Assistant was designed to guide medical students in their preparation for Ocupational English Test which represents a major obstacle in their way of starting their medical career. "
    "The chatbot for OET exam preparation provides guidance on exam structure, offers practice questions with feedback "
    "shares language tips and strategies, aids in vocabulary building, assists with grammar and syntax"
    "facilitates speaking and writing practice, tracks progress, recommends resources, addresses FAQs"
    
    "This AI App was developed by <b>MOHAMMED BAHAGEEL</b>, Artificial intelligence scientist as a part of his experiments using Retrieval Augmented Generation."
    "</div>",
    unsafe_allow_html=True
)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="")]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store()

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human", avatar="👨‍⚕️"):
        st.markdown(user_query)

    with st.chat_message("AI",avatar="🤖"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))
