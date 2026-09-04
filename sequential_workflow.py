from typing import TypedDict
from langchain_mistralai import ChatMistralAI
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, START, END


class MyState(TypedDict):
    raw_text : str
    edited_text: str
    script_text : str
    translated_text : str

model = ChatGroq(
    model = "groq/compound",
    temperature = 0.8 
)

def editor_node(State:MyState)-> dict:
    """Stage 1: Cleans up grammar, removes typos, and refines the tone."""

    raw_text = State['raw_text']

    prompt =  (
        "You are an expert copyeditor. Clean up the following raw text. "
        "Fix any grammatical errors, spelling mistakes, and smooth out the transition flow "
        "while keeping the core message intact. Return only the edited text.\n\n"
        f"Text:\n {raw_text}"
    )

    result = model.invoke(prompt)

    return {"edited_text": result.content}


def scriptwriter_node(State:MyState)-> dict:
    """Stage 2: Formats the clean text into an engaging video script style."""

    Edited_text = State['edited_text']

    prompt = (
        "You are a charismatic YouTube content creator. Take this edited text and transform "
        "it into a highly engaging, punchy, conversational video script hook. Make it sound "
        "like a real person speaking passionately. Return only the script content.\n\n"
        f"Edited Text:\n{Edited_text}"
    )
    result = model.invoke(prompt)

    return {"script_text": result.content}


def translator_node(State:MyState)-> dict:
    """Stage 3: Translates the script into natural flowing Hinglish."""

    prompt = (
        "You are an expert content localizer for the Indian market. Take the following script "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
        "or repeat information. Alternating comfortably between Hindi and English phrases just like "
        "an intellectual tech educator would speak naturally on a live stream. Keep the energy high! "
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{State['script_text']}"
    )

    result = model.invoke(prompt)

    return {"translated_text": result.content}

graph = StateGraph(MyState)

graph.add_node("editor_node", editor_node)
graph.add_node("scriptwriter_node",scriptwriter_node)
graph.add_node("translator_node", translator_node)

graph.add_edge(START, "editor_node")
graph.add_edge("editor_node","scriptwriter_node")
graph.add_edge("scriptwriter_node", "translator_node")
graph.add_edge("translator_node", END)

app = graph.compile()         # makes the graph a rummable, now we can invoke the graph.

result = app.invoke({
    "raw_text": "AI agents are the future of tech. They can think, plan, and act on their own. LangGraph helps you build these agents with proper control and memory." 
})

print(result['translated_text'])