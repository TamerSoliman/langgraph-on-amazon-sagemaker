"""
Multi-Agent System - Agent Modules

This package contains specialized agents for the multi-agent workflow.

For AI/ML Scientists:
Each agent is like a specialized model in an ensemble - they have different
roles and work together to solve complex tasks.
"""

from .researcher import researcher_node
from .writer import writer_node
from .reviewer import reviewer_node, should_revise

__all__ = [
    'researcher_node',
    'writer_node',
    'reviewer_node',
    'should_revise'
]
