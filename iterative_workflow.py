
from typing import TypedDict, Annotated
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
# pyrefly: ignore [missing-import]
from langchain_tavily import TavilySearch
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END 
from langgraph.prebuilt import ToolNode

tavily_search = TavilySearch(max_results = 3)
tools = [tavily_search]

writer_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature = 0.8
    )

writer_llm_with_tools = writer_llm.bind_tools(tools)

reviewer_llm = ChatGroq(
    model = "groq/compound",
    temperature = 0.1
    )



class MyState(TypedDict):
    topic:str
    messages : Annotated[list, add_messages]
    draft:str
    is_approved:bool
    attempt:int
    reviewer_feedback:str

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)

def writer_node(state:MyState)->dict:
    """Writes (or rewrites) the LinkedIn post. Can call Tavily to search first."""
    topic = state["topic"]
    attempt = state.get("attempt",0) + 1
    previous_feedback = state.get("reviewer_feedback", "")

    if(attempt==1):
        user_prompt = (
            f"Write a LinkedIn post on this topic {topic}"
            f"if you need current info search the web first "
        )
    else:
        user_prompt = (
            f"your previous draft on '{topic}' was rejected"
            f"Here is the reviewer's feedback \n\n {previous_feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentiond"
            f"do not repeat the same mistake"
        )
    
    query = [("system", WRITER_SYSTEM_PROMPT),("human", user_prompt)]

    response = writer_llm_with_tools.invoke(query)

    return {
        "draft" : response.content,
        "messages" : [("human", user_prompt), response],
        "attempt" : attempt
    }

tool_node = ToolNode(tools)



REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)

def reviewer_node(state:MyState)->dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    draft = state["draft"]
    print(f"\n\n generated post \n {draft} \n ")

    prompt = (
        f"review this LinkedIn post draft : \n"
        f"{draft}\n"
        f"give your reviews"
    )

    query = [("system", REVIEWER_SYSTEM_PROMPT), ("human", prompt)]

    response = reviewer_llm.invoke(query)

    review = response.content.strip()
    review_array = review.upper().split("FEEDBACK:")

    if("REJECTED" in review_array[0]):
        is_approved = False
    else:
        is_approved = True

    if(is_approved):
        feedback = ""
    else:
        feedback = review_array[1]
    
    verdict = "APPROVED" if is_approved else "REJECTED"
    print(f"[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]")

    return{
        "is_approved" : is_approved,
        "reviewer_feedback" : feedback
    }

    
def router_at_writer(state:MyState)->str:
    last_message = state['messages'][-1]
    tool_calls = last_message.tool_calls
    if(tool_calls):
        return "tool_node"
    else:
        return "reviewer"


def router_at_reviewer(state:MyState)->str:
    is_approved = state['is_approved']
    
    if(is_approved):
        print("post haas been approved \n")
        return END
    if state['attempt'] >= 3:
        print("reached max attempts")
        return END 
    
    return "writer"


graph = StateGraph(MyState)

graph.add_node("writer", writer_node)
graph.add_node("reviewer", reviewer_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START,"writer")
graph.add_conditional_edges("writer", router_at_writer)
graph.add_edge("tool_node","writer")
graph.add_conditional_edges("reviewer", router_at_reviewer)

app = graph.compile()



print("=" * 55)
print("Welcome to the LinkedIn Post Generator")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, review it")
print("itself, and iterate until it's publish-ready.")

print("=" * 55)

topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

if not topic:
    print("\nNo topic given. Exiting.")
else:
    print("\nStarting generation...\n")

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "reviewer_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final_state["draft"])
    print("=" * 55)
    print(f"Total attempts: {final_state['attempt']}")
    print(f"Approved: {final_state['is_approved']}") 