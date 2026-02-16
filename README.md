LM markdown for education is a system that allows faculty to record their contents and insights on a topic for sharing with others. While primarily developed to support teaching, it can more generally store one's own experience in a fromat that can be retold by large language models.

The vision is to craft a system where, starting from one own's notes, pdfs, or videos, or from authoring one's own markdown, one can produce an interactive chatbot with one own's content, or interactive videos of one's lectures.

To install the system after cloning it from github, use poetry:

```bash
poetry install
```

The documentation can be displayed in a web browser. Type

```bash
poetry shell
mkdocs serve
```

The project uses settings that can be configured through the files config.toml and appchat.toml. Using the programme (such as starting the web server) will at some point create these files with default values, so that they can be further modified. These files can also be created explicitly from the terminal:

```bash
poetry shell
lmme create-default-config-file
```

NOTE: You will need the LLM API keys in your environment variables to use the chatbot.