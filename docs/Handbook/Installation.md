# Installation and quickstart

LM Markdown for Education is still in the early development phase. For this reason, the installation follows the steps you would take for installing a coding project.

## Start a new project

To start a new project, plan a new folder (for those who use git, they might want to create a new repository). This folder will contain the local copy of the LM Markdown for Education software and your source material.

The first step is to install the LM Markdown for Education software. You need the following software installed on your computer:

- Python, at best in version 3.13, but not older than 3.12. LM Markdown for Education will not work on newer versions of Python such as 3.14.
- poetry. See https://python-poetry.org/docs/#installing-with-the-official-installer for instructions.
- git. See https://git-scm.com/install/.

Navigate with your terminal in the folder that will contain the new folder, then clone the code from git:

```bash
git clone https://github.com/roberto-viviani/lmm-education.git
```

Git will create the folder with the code inside. Go into this folder, and install the software using Poetry:

```bash
cd lmm-education
poetry install
```

Poetry will check that you have a Python version compatible with the project. If Poetry complains that it cannot find a compatible version, install Python 3.13 manually or with Poetry:

```bash
poetry python install 3.13
```

It commonly happens that `poetry install` requires you to run `poetry lock`. You may get an error message like this,

```bash
$ poetry install
The currently activated Python version 3.11.4 is not supported by the project (>
=3.12.0,<3.14.0).
Trying to find and use a compatible version.
Using python.exe (3.13.12)
Creating virtualenv lmm-education in D:\Scratch_Roberto\R\lmm-education\.venv
Installing dependencies from lock file

pyproject.toml changed significantly since poetry.lock was last generated. Run `
poetry lock` to fix the lock file.
```

This does not mean that there is any fundamental problem with the installation, but merely that the dependencies have to be regenerated. Repeat the install after giving the `lock` command:

```bash
poetry lock
poetry install
```

As in all Python project, you now have to activate the environment. Use the command `poetry env activate` to discover how to do that (alas, this tends to be a bit platform specific). In some circumstances, you might get away with this by simple giving the command

```bash
poetry shell
```

Now you can start using the application. You can check that everything is running by operating the CLI:

```bash
lmme --help
```

which gives you an overview of available commands.

## Working on documents

It is a good idea to set up another folder within this main folder to contain your documents, but this is not absolutely necessary. A possible setup is to use the main folder to work on an active document, and save the documents that you want to make up your chatbot in a dedicated sub-folder (for example, .sources/).

If you already have a set of documents, copy them in this folder, and go to [RAG authoring](RAGauthoring.md) to get started.

## Configuring the application

Details on how to configure the chatbot are in [Configuration](Configuration.md).

## Starting the chatbot

Instructions for starting the chatbot are in [ChatApp](ChatApp.md).

