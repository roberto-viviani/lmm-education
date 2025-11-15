from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .config import ServerSettings

CHAT_CONFIG_FILE: str = "appchat.toml"


class CheckResponse(BaseSettings):
    """
    Settings to check appropriateness of chat.
    """

    check_response: bool = Field(default=False)
    allowed_content: list[str] = Field(default=[])

    @model_validator(mode='after')
    def validate_allowed_content(self):
        """Validate that allowed_content is not empty when check_response is True."""
        if self.check_response and not self.allowed_content:
            raise ValueError(
                "allowed_content must not be empty when check_response is True"
            )
        return self


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
    comment: str = Field(
        default="Please leave a comment on the response of the chatbot here"
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

    # thematic control of interaction
    check_response: CheckResponse = Field(
        default_factory=CheckResponse,
        description="Check thematic appropriateness of chat",
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
