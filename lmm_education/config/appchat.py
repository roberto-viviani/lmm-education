from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .config import ServerSettings

CHAT_CONFIG_FILE: str = "appchat.toml"


class ChatSettings(BaseSettings):
    # This is displayed on the chatbot. Change it as appropriate
    title: str = Field(default="VU Study Assistant")
    description: str = Field(
        default="""
Study assistant chatbot for VU Specific Scientific Methods 
in Psychology: Data analysis with linear models in R. 
Ask a question about the course, and the assistant will provide a 
response based on it. 
Example: "How can I fit a model with kid_score as outcome and mom_iq as predictor?" 
"""
    )

    # messages
    MSG_EMPTY_QUERY: str = Field(
        default="Please ask a question about the course."
    )
    MSG_WRONG_CONTENT: str = Field(
        default="I can only answer questions about the course."
    )
    MSG_LONG_QUERY: str = Field(
        default="Your question is too long. Please ask a shorter question."
    )
    MSG_ERROR_QUERY: str = Field(
        default="I am sorry, I cannot answer this question. Please retry."
    )

    SYSTEM_MESSAGE: str = Field(
        default="""
You are a university tutor teaching undergraduates in a statistics course 
that uses R to fit models, explaining background and guiding understanding. 
Please assist students by responding to their QUERY by using the provided CONTEXT.
If the CONTEXT does not provide information for your answer, integrate the CONTEXT
only for the use and syntax of R. Otherwise, reply that you do not have information 
to answer the query.
"""
    )

    PROMPT_TEMPLATE: str = Field(
        default="""
Please answer my QUERY by using the provided CONTEXT. 
Please answer in the language of the QUERY.
---
CONTEXT: "{context}"

---
QUERY: "{query}"

---
RESPONSE:

"""
    )

    server: ServerSettings = Field(
        default_factory=ServerSettings,
        description="Server configuration",
    )

    model_config = SettingsConfigDict(
        toml_file=CHAT_CONFIG_FILE,
        env_prefix="LMMEDU_",  # Uppercase for environment variables
        frozen=False,
        validate_assignment=True,
        extra='forbid',  # Prevent unexpected fields
    )
