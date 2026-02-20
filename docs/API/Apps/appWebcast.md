# appWebcast

The appWebcast module is the entry point for the video-displaying RAG chatbot in LM Markdown for Education. By coding at the console,

```bash
appWebcast
```
one starts the chatbot server at the port specified in the appchat.toml configuation file.

The appWebcast.py module itself consists of three parts. After loading and initializing the required libraries in the first part, the module codes the callback functions that determine the behaviour of the web page when the user interacts with it (see functions below). In the last part, the Gradio interface is coded.

::: lmm_education.appWebcast
    options:
      show_root_heading: false