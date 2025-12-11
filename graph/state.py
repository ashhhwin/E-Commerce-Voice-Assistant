from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """
    State container for agent workflow execution.
  
    Attributes:
        audio_path: File path to input audio recording
        transcript: Transcribed text from audio input
        intent: Parsed user intent with extracted parameters
        plan: Execution plan with steps and strategy
        evidence: Retrieved supporting data organized by source
        answer: Generated response text
        citations: Reference metadata for answer sources
        safety_flags: List of detected safety or content warnings
        tts_path: File path to generated text-to-speech audio
        log: Execution trace with timestamps and events
    """
    audio_path: str | None
    transcript: str | None
    intent: Dict[str, Any] | None
    plan: Dict[str, Any] | None
    evidence: Dict[str, List[Dict]] | None
    answer: str | None
    citations: List[Dict] | None
    safety_flags: List[str] | None
    tts_path: str | None
    log: List[Dict] | None