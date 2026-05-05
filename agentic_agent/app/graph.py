# app/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.state import PatientState
# from app.nodes.audio_input import record_audio
from app.nodes.transcribe import transcribe_audio
from app.nodes.extract_symptoms import extract_symptoms
from app.nodes.followup import followup_node   
from app.nodes.approval import approval_node     
from app.nodes.run_mri import run_mri_node
from app.nodes.summary import summary_node
from app.nodes.doctor_review import doctor_review_node
from app.nodes.save_results import save_results_node

def route_after_followup(state: PatientState) -> str:
    if state.status == "ready_for_approval":
        return "doctor_approval"
    return "followup"   # NodeInterrupt will pause here each round

def _route_after_approval(state: PatientState) -> str:
    """
    Called after doctor_approval.

    Reads two state flags set by approval_node:
      - state.doctor_approved  => doctor signed off on the symptom notes
      - state.mri_requested    => doctor also flagged that an MRI is needed

    Returns the name of the next node to run.
    """
    if state.doctor_approved and state.mri_requested:
        return "run_mri"       # run_mri (will interrupt until scan is uploaded)
    return "save_results"      # save and end (no MRI needed or not approved)


builder = StateGraph(PatientState)

# builder.add_node("record_audio",     record_audio)
# builder.add_node("transcribe",       transcribe_audio)
builder.add_node("transcribe",       transcribe_audio)
builder.add_node("extract_symptoms", extract_symptoms)  
builder.add_node("followup",         followup_node)    
builder.add_node("doctor_approval",  approval_node)    
builder.add_node("run_mri",          run_mri_node)
builder.add_node("summary",          summary_node)
builder.add_node("doctor_review",    doctor_review_node)
builder.add_node("save_results",     save_results_node)

# builder.add_edge(START,              "record_audio")
# builder.add_edge("record_audio",     "transcribe")
builder.add_edge(START,              "transcribe")
builder.add_edge("transcribe",       "extract_symptoms")
builder.add_edge("extract_symptoms", "followup")
# builder.add_edge("followup",         "doctor_approval")
builder.add_conditional_edges(
    "followup",
    route_after_followup,
    {
        "doctor_approval": "doctor_approval",
        "followup":        "followup",        # loops back to itself - interrupted each time
    }
)

# conditional branch after doctor approval
builder.add_conditional_edges(
    "doctor_approval",
    _route_after_approval,
    {
        "run_mri":      "run_mri",
        # "summary": "summary",
        "save_results": "save_results"
    },
)

# run_mri => summary => doctor_review => save
builder.add_edge("run_mri",       "summary")
builder.add_edge("summary",       "doctor_review")   # always requires review
builder.add_edge("doctor_review", "save_results")
builder.add_edge("save_results",  END)

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)