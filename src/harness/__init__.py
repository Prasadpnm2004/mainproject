"""
harness — Agent execution host package.

Provides modular agent harnesses that wrap different agent frameworks
(LangGraph in Session 1; custom minimal loop and AutoGen in later sessions).
Each harness exposes a uniform interface so experiments in experiments/ can
swap frameworks without changing their own code.
"""
