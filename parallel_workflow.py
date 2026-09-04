
from typing import TypedDict, Annotated
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, START, END


model = ChatMistralAI(
    model="mistral-small-latest",
)

def merge_dict(old:dict, new:dict):
    if old is None:
        return {**new}
    return {**old, **new}

class myState(TypedDict):
    raw_text:str
    safety_score: Annotated[dict[str,int] , merge_dict]

def toxicity_node(state:myState)->dict:

    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    result = model.invoke(prompt)

    try:
        score = int(result.content.strip())
    except ValueError:
        score = 0

    return {"safety_score": {"toxicity_score" : score}}


def copyright_node(state:myState)->dict:
    print("\n🔏 [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n {state['raw_text']}"
    )

    result = model.invoke(prompt)

    try:
        score = int(result.content.strip())
    except ValueError:
        score = 0

    return {"safety_score" : {"copyright_score" : score}}


def cultural_node(state:myState)->dict:

    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    result = model.invoke(prompt)

    try:
        score = int(result.content.strip())
    except ValueError:
        score = 0

    return {"safety_score" : {"cultural_score" : score}}


#creating graph 
graph = StateGraph(myState)

#creating nodes
graph.add_node("toxicity_node",toxicity_node)
graph.add_node("copyright_node",copyright_node)
graph.add_node("cultural_node",cultural_node)

# creating edges
graph.add_edge(START,"toxicity_node")
graph.add_edge(START,"copyright_node")
graph.add_edge(START,"cultural_node")

graph.add_edge("toxicity_node",END)
graph.add_edge("copyright_node",END)
graph.add_edge("cultural_node",END)

# compiling graph
app = graph.compile()


sample_script = {"raw_text": """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into 
    your friend's system using a script I copied directly from an online forum. 
    Honestly, traditional security protocols are absolute garbage and anyone still using 
    them is an absolute idiot. Let's dive into the code!
    """}

result = app.invoke(sample_script)

print(result['safety_score'])