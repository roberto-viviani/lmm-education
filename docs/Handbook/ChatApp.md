# Configuring and starting the chat app

To start the chat app, open a console in the folder where LM Markdown for Education was installed, and activate the Python environent. Then start the app:

```bash
python -m appChat
```

The app will print on the console a list of messages like the following

```bash
appchat.toml created in app folder, change as appropriate
INFO - Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
* Running on local URL:  http://127.0.0.1:7860
INFO - HTTP Request: GET http://127.0.0.1:7860/gradio_api/startup-events "HTTP/1.1 200 OK"
INFO - HTTP Request: HEAD http://127.0.0.1:7860/ "HTTP/1.1 200 OK"
* To create a public link, set `share=True` in `launch()`.
INFO - HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"
INFO - HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"
```

If everything is working, you have the app running locally at http://127.0.0.1:7860/. You can close the app by pressing Ctrl-C, and look at appchat.toml file that the app has created in the app folder. This file contains a template to customize the app, initialized to default values.

There are three types of settings that you will probably want to change: the title and description that appear on the website, the system prompt or perhaps also other prompts, and the configuration of the server.

## Title and description

The title and description should be changed to reflect the course you are teaching. The defaults refer to a statistics course. Replace them with new textual values, not forgetting to include them within the quotes (").

```ini
title = "VU Study Assistant"
description = "\nStudy assistant chatbot for VU Specific Scientific Methods \nin Psychology: Data analysis with linear models in R. \nAsk a question about the course, and the assistant will provide a \nresponse based on it. \nExample: \"How can I fit a model with kid_score as outcome and mom_iq as predictor?\" \n"
```

## Prompts

You will probably also want to change the system prompt. The other messages that you see in this part of the configuration file may not need to be changed, unless you want to offer the app in a language other than English.

```ini
SYSTEM_MESSAGE = "\nYou are a university tutor teaching undergraduates in a statistics course \nthat uses R to fit models, explaining background and guiding understanding. \nPlease assist students by responding to their QUERY by using the provided CONTEXT.\nIf the CONTEXT does not provide information for your answer, integrate the CONTEXT\nonly for the use and syntax of R. Otherwise, reply that you do not have information \nto answer the query.\n"
```

## Server settings

The server settings are in the server section:

```ini
[server]
mode = "local"
port = 61543
host = "localhost"
```

If you want to offer the server over the internet, change `local` into `remote`. You can also change the port to which the app is listening. The host parameter is not used at present.

### Instructing the server to check the chat content

You can instruct the language model to check that the chat is taking place within the topic of the course. LM Markdown for Education uses a secondary language model to classify the response of the model before releasing it to the chat. In the following, the chat is configured to be limited to statistics and software programming.

```ini
[check_response]
check_response = true
allowed_content = ['statistics', 'software programming']
```

Put your topics within the square brackets. For example, if you have only one topic, code `allowed_content = ['statistics']`.

## Other settings

If you have not used LM Markdown for Education before starting the app, it will also have written a second configuration file with default settings, config.toml. This file contains settings that are common to the app and the rest of LM Markdown for Education (for example, when interacting with it through the CLI). You may want to configure this file too (see [configuration](Configuration.md) for details).