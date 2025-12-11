import logging
from langgraph.graph import StateGraph, END
from .state import AgentState
from .agents.intent_parser import route
from .agents.query_strategist import plan
from .agents.data_fetcher import retrieve
from .agents.response_synthesizer import answer
from .agents.quality_validator import critique

logger = logging.getLogger(__name__)

INTENT_CLASSIFIER = "intent_classifier"
QUERY_PLANNER = "query_planner"
DATA_RETRIEVER = "data_retriever"
RESPONSE_GENERATOR = "response_generator"
QUALITY_EVALUATOR = "quality_evaluator"


def create_workflow():
    """
    Construct and compile the agent workflow graph.
    
    Builds a sequential processing pipeline: intent classification → query planning →
    data retrieval → response generation → quality validation.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    graph = StateGraph(AgentState)
    
    graph.add_node(INTENT_CLASSIFIER, route)
    graph.add_node(QUERY_PLANNER, plan)
    graph.add_node(DATA_RETRIEVER, retrieve)
    graph.add_node(RESPONSE_GENERATOR, answer)
    graph.add_node(QUALITY_EVALUATOR, critique)
    
    graph.set_entry_point(INTENT_CLASSIFIER)
    graph.add_edge(INTENT_CLASSIFIER, QUERY_PLANNER)
    graph.add_edge(QUERY_PLANNER, DATA_RETRIEVER)
    graph.add_edge(DATA_RETRIEVER, RESPONSE_GENERATOR)
    graph.add_edge(RESPONSE_GENERATOR, QUALITY_EVALUATOR)
    graph.add_edge(QUALITY_EVALUATOR, END)
    
    logger.info("Agent workflow graph compiled successfully")
    
    return graph.compile()