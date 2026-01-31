"""Creates LangGraph workflows."""

from lmm.utils.logging import LoggerBase, ConsoleLogger
from lmm_education.config.config import ConfigSettings, load_settings

from .langchain.base import ChatStateGraphType
from .langchain.chat_graph import create_chat_workflow
from .langchain.chat_agent import create_chat_agent


def workflow_factory(
    workflow_name: str,
    settings: ConfigSettings | None = None,
    logger: LoggerBase = ConsoleLogger(),
) -> ChatStateGraphType:
    """
    Factory function to retrieve compiled workflow graphs.

    Args:
        workflow_name: the name of the workflow to load.
        settings: a ConfigSettings object for the major,
            minor, and aux models used in the workflow
        logger: a logger object for errors.

    Behaviour:
        raises errors if fails to load settings or create workflow.
    """

    # will raise errors if failed.
    if settings is None:
        settings = load_settings(logger=logger)
    if settings is None:
        raise ValueError("Could not create workflow.")

    # At present, we only have one graph, so we put this function
    # here, but it will be moved to a factory module when we have
    # more workflows.
    match workflow_name:
        case "workflow":  # only query at first chat
            return create_chat_workflow(settings)
        case "agent":
            return create_chat_agent(settings)
        case _:
            raise ValueError(f"Invalid workflow: {workflow_name}")
