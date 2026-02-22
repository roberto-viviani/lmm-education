"""
LangGraph-based chat workflows for RAG-enabled language model
interactions.

This package implements the core workflow engine for the application's
chatbot. It is defined by three things: graph definition, graph 
streaming, and dependency injection.


1. Graphs and State
====================

A graph in LangGraph is a directed graph of **nodes** (Python functions
that transform a shared state dictionary) connected by **edges** (which
determine execution order). At each step, the node receives the current
state and returns a partial update to it; LangGraph merges the update
back into the state.

The shared state is defined as a ``TypedDict`` called ``ChatState``
(see ``base.py``). It carries the user query, conversation history
(as LangChain messages), the retrieved context, the generated response,
timing information, and a ``status`` field that drives routing
decisions. A graph is initialised with ``create_initial_state()``,
which populates the required fields.

Two concrete graphs are provided:

- **chat_graph.py** — a linear workflow: validate → integrate history
  → retrieve context → format query → generate response.
- **chat_agent.py** — an agentic workflow where the LLM decides when
  to call a ``retrieve_context`` tool, allowing multi-hop retrieval
  and iterative reasoning.

Both are compiled into a ``CompiledStateGraph`` (aliased as
``ChatStateGraphType``) by their respective ``create_chat_workflow()``
and ``create_chat_agent()`` factory functions. The ``workflow_factory``
module (one level up, ``workflows/workflow_factory.py``) selects between
them at runtime based on the ``appchat.toml`` configuration.


2. Shared Nodes and Routing
============================

Nodes shared by both graphs — ``validate_query`` and
``integrate_history`` — live in ``nodes.py`` so that they have a
single definition. Because ``integrate_history`` needs to capture
``ConfigSettings`` for the summariser model, it is produced by a
**factory** function ``create_integrate_history_node(settings)``.

Conditional routing functions (``continue_if_valid``,
``continue_if_no_error``, ``continue_after_tool_call``) live in
``graph_routing.py``. They return callbacks that inspect the
``status`` field of ``ChatState`` to decide whether the graph should
continue to the next node or terminate.


3. Dependency Injection
========================

Graph nodes receive external resources through LangGraph's **runtime
context** mechanism. A ``ChatWorkflowContext`` (defined in ``base.py``)
is a Pydantic model that carries:

- ``retriever`` — the vector-store retriever (``BaseRetriever``)
- ``system_message`` — the system prompt
- ``chat_settings`` — the ``ChatSettings`` from ``appchat.toml``
- ``logger`` — a ``LoggerBase`` instance

The context is constructed once — typically from the configuration
files via ``ChatWorkflowContext.from_default_config()`` — and passed
to the graph at stream time. Nodes access it through
``runtime.context``, keeping them free of global state and easy to
test in isolation.


4. Streaming Architecture
==========================

Graphs are designed to be consumed as asynchronous streams, not via
``.ainvoke()``. The streaming infrastructure is organised in
``stream_adapters.py`` as a **three-tier** pipeline of composable
async iterators:

Tier 1 — Multi-mode raw stream
    The richest representation, produced by calling
    ``stream_graph_state()`` or ``stream_graph_updates()``. Each
    item is a ``(mode, event)`` tuple, where mode is "messages",
    "values", or "updates". This tier carries everything: message
    chunks for display, full state snapshots for logging, and
    differential updates for reacting to specific field changes.

Tier 2 — Messages only
    Produced by ``tier_1_to_2_adapter()``. Each item is a
    ``(chunk, metadata)`` tuple from the "messages" stream mode.
    State and update events are discarded.

Tier 3 — Plain text
    Produced by ``tier_2_to_3_adapter()`` or ``tier_1_to_3_adapter()``.
    Each item is a ``str`` — the text fragment to display.

Information flows downward only: tier 1 → tier 2 → tier 3. Adapters
at each tier can filter, transform, or inject content. Key adapters
include:

- ``terminal_tier1_adapter`` — calls a callback with the terminal
  state (used to log the exchange to a database after the stream
  completes).
- ``tier_1_filter_messages_adapter`` — suppresses messages from
  specific nodes (e.g., tool calls in the agent workflow).
- ``field_change_tier_1_adapter`` — reacts to state field changes
  (e.g., printing retrieved context).
- ``terminal_field_change_adapter`` — a tier 1 → tier 3 shortcut
  that emits text and additionally inserts output when specific
  state fields change.

The domain-specific ``stateful_validation_adapter`` in
``chat_stream_adapters.py`` is a tier 1 adapter that buffers streamed
chunks, sends them to a secondary LLM for content classification,
and suppresses the stream if the content falls outside the allowed
categories.

A typical composition in ``query.py`` looks like::

    raw = stream_graph_state(workflow, initial_state, context)
    validated = stateful_validation_adapter(raw, ...)
    logged = terminal_tier1_adapter(validated, on_terminal_state=...)
    text = tier_1_to_3_adapter(logged)

    async for chunk in text:
        print(chunk, end="")


5. Database Logging
====================

``graph_logging.py`` defines ``ChatDatabaseInterface`` (an ABC) and
two concrete CSV-based implementations (``CsvChatDatabase`` for
in-memory streams, ``CsvFileChatDatabase`` for files). The
``graph_logger()`` function in ``base.py`` extracts fields from
``ChatState`` and ``ChatWorkflowContext`` and delegates to the
database interface. It is invoked as a terminal-state callback
through ``terminal_tier1_adapter`` in the stream pipeline.


6. Module Map
==============

Foundation
    ``base.py``
        ``ChatState``, ``ChatWorkflowContext``,
        ``create_initial_state()``, ``prepare_messages_for_llm()``,
        ``graph_logger()``.

Graph Definition
    ``nodes.py``
        Shared node functions: ``validate_query``,
        ``create_integrate_history_node``.
    ``graph_routing.py``
        Conditional-edge callbacks: ``continue_if_valid``,
        ``continue_if_no_error``, ``continue_after_tool_call``.
    ``chat_graph.py``
        Linear RAG workflow — ``create_chat_workflow()``.
    ``chat_agent.py``
        Agentic RAG workflow — ``create_chat_agent()``.

Streaming
    ``stream_adapters.py``
        Generic three-tier stream adapters, entry-point functions
        (``stream_graph_state``, ``stream_graph_updates``).
    ``chat_stream_adapters.py``
        Domain-specific adapter: ``stateful_validation_adapter``.

Logging
    ``graph_logging.py``
        ``ChatDatabaseInterface``, ``CsvChatDatabase``,
        ``CsvFileChatDatabase``.
"""
