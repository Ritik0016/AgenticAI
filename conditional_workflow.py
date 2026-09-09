from typing import TypedDict, Annotated
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph.message import add_messages

from dotenv import load_dotenv
load_dotenv()

#building models
chat_model = ChatGroq(
    model = "groq/compound",
    temperature = 0.4
)
embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

#state
class State(TypedDict):
    programme: str
    messages: Annotated[list,add_messages]
    query_type: str
    retrived_context: str

# retrivers & vector store
def build_retrivers(file_path:str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 100)
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embedding_model)

    retriever = vector_store.as_retriever(search_kwargs = {"k" : 4})

    return retriever

academic_retriever = build_retrivers("academics_handbook.pdf")
fees_retriever = build_retrivers("fee_structure.pdf")


# finds which node is the best fit according to the question asked by the user
def classifier_node(state:State)->dict:
    """Look at the latest user message and decide which path to take."""

    query = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {query}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = chat_model.invoke(prompt)
    category = response.content.strip().lower()

    if("academic" in category):
        category = "academic"
    elif("fee" in category):
        category = "fee"
    else:
        category = "general"

    return {
        "query_type" : category
    }

# this is actually use for routing 
def router(state:State)->str:
    if(state["query_type"] == "academic"):
        return "academic_node"
    elif(state["query_type"] == "fee"):
        return "fee_node"
    else:
        return "general_node"

# academic node
def academic_node(state:State)->dict:

    query = state["messages"][-1].content
    docs = academic_retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)
    return {"retrived_context" : context}

# fee_node
def fee_node(state:State)->dict:
    query = state["messages"][-1].content
    docs = fees_retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)
    return {"retrived_context" : context}

# general node
def general_node(state:State)->dict:
    return{"retrived_context" : "NO CONTEXT RETRIVED"}


# response node
def response_node(state:State)->dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state['programme']
    context = state["retrived_context"]

    

    if(context == "NO CONTEXT RETRIVED"):
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
        response = chat_model.invoke(prompt)

    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )
        response = chat_model.invoke(prompt)

    return {
        "messages": [("ai", response.content)]
    }

# creating graph 
graph = StateGraph(State)

# nodes
graph.add_node("classifier_node",classifier_node)
graph.add_node("academic_node",academic_node)
graph.add_node("fee_node",fee_node)
graph.add_node("general_node",general_node)
graph.add_node("response_node",response_node)

# edges
graph.add_edge(START,"classifier_node")
graph.add_conditional_edges("classifier_node", router)
graph.add_edge("academic_node","response_node")
graph.add_edge("fee_node", "response_node")
graph.add_edge("general_node","response_node")
graph.add_edge("response_node",END)

app = graph.compile()
