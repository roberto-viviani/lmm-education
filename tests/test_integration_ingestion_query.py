"""Test ingestion and retrieval"""

import unittest

document = """
---
author: Roberto Viviani
date: '2025-05-04'
docid: Ch1
title: Chapter 1
summary: These lectures review linear models from a practical perspective, focusing on understanding over programming and using R to specify and fit models. They highlight the importance of linear models in analyzing predictor-outcome relationships, addressing confounding in observational studies, and enabling predictions. The approach emphasizes aligning with professional statistical practices and accessing advanced data analysis techniques.
titles: Chapter 1
frozen: True
---

---
summary: Linear models are widely used in practical statistics to analyze relationships between variables. They model the connection between predictors (independent variables) and an outcome (dependent variable), often visualized as a line relating a predictor to its outcome. While simple linear models focus on this direct relationship, more complex models capture intricate associations. These models are primarily used to assess the significance of predictor-outcome relationships, predict outcomes based on predictor values, and interpret the influence of variables. However, they do not establish causality and can be affected by confounding factors, especially in observational data. The interpretation of linear models relies on understanding the relationship between predictors and outcomes, with their limitations rooted in their focus on numerical associations rather than real-world causality. Despite their limitations, linear models are essential tools for analyzing and predicting data, with their effectiveness depending on the context and the assumptions made.
~txthash: S/ec1NIX0xV3NhFFuHY6bQ
titles: Chapter 1 - What are linear models?
source: Ch1
questions: "What are the two main ways linear models are used in practice? - How do linear models relate predictors to an outcome variable? - Why is understanding the output of linear models important for their correct application?  - How do generalizations of linear models enable capturing more complex associations? - In what way does the purpose of a linear model (prediction vs. inference) affect the consideration of confounding factors?"
---
## What are linear models?

Linear models and their generalizations constitute the majority of the statistical models used in practice. Here, we will look at linear models from a practical perspective, emphasizing the issues in correctly applying them and in understanding their output.

Linear models capture the association between a set of variables, the *predictors* (also known as *independent variables*), and an *outcome* variable (or *dependent variable*). In the simplest setting, this association may be approximated and displayed by a line relating a predictor and the outcome. However, the generalizations of linear model allow capturing much more complex associations, as we shall see.

There are two broad ways of using linear models. In the first, which is perhaps the most common, we are interested in assessing the relationship between predictors and the outcome in order to establish if this relationship is "significant".

The second use of linear models is to predict the outcome given certain values of the predictors. This use of linear models is the same as in machine learning. In this case, after the fit of a linear model has been computed, one may use values of the predictors that the model had not seen to predict the outcome.

---
summary: This text discusses the importance of understanding the relationship between predictors and outcomes in observational and experimental studies. It emphasizes that in observational research, confounding factors—variables that influence both predictors and outcomes—can distort results, especially when they are related to socioeconomic conditions or other third variables. Randomization in experimental studies helps mitigate confounding by equalizing these factors across groups, making causal inferences more reliable. The summary highlights that while models can predict outcomes, they do not establish causality, and confounding remains a challenge, particularly in observational data. Ultimately, the way models are used and interpreted affects the validity of conclusions about predictor-outcome relationships.
~txthash: 7JdFs+GLpXTfvmONOnyc5g
titles: Chapter 1 - What are linear models? - Observational and experimental studies
source: Ch1
questions: "What are observational studies? - What is the difference between observational and experimental studies in assessing the significance of associations?  - How does confounding affect the interpretation of associations in observational studies? - Why can causal inferences be more confidently made from experimental studies compared to observational studies? - In what context do the issues of confounding and causality become less relevant, according to the text? - How does the purpose of the model (prediction vs. inference) influence the importance of confounding factors?"
---
### Observational and experimental studies

We will first discuss issues arising from assessing the significance of associations.

When we look at the significance of the association between predictors and outcomes, it is important to distinguish between two different settings. In the first, we have variables that we have observed in the field. For example, early traumas may exert an influence on the predisposition to psychopathology in adult age. We do not have any way to change the occurrence of traumas in the past with an experiment, so we look at their consequences in a sample from the population. Studies of this type are called *observational*. An important issue in observational studies is that predictors may be *confounded* by other factors affecting the outcome. Confounding occurs when the relationship between the predictor and the outcome is distorted by the presence of a third variable. For example, traumas may occur more frequently in adverse socioeconomic conditions, and these conditions may in turn adversely affect the predisposition to psychopathology. When we assess the association between traumas and psychopathology, the association we find may inadvertently include the effects of socioeconomic conditions, as traumatized individuals are often those who are most disadvantaged socioeconomically.

In the second setting, the predictor of interest is a variable representing an experimental manipulation. Examples include treatment conditions, changes in aspects of a cognitive task, or different levels of a stimulus. A key aspect of such *experimental* studies is that the value of the predictor of interest, being determined by the experimenter, can be randomized. For this reason, at least in the long term, the action of other possible variables on the outcome cancels out, as there is no relationship between these potential confounders and the predictor of interest. We cannot randomize traumas, but if we could, we would no longer have the potential confounder of adverse socioeconomic conditions, because randomization ensures that, in the long run, individuals from all conditions are equally traumatized.

Importantly, the way in which we estimate models in observational and experimental studies is exactly the same. However, the conclusions that we may draw from establishing the likely existence of an association between predictor and outcome differ in these two cases. In observational studies, we can only infer the mere association, unless a sometimes fairly extensive efforts are made on being able to contain the effect of confounders. In experimental studies, we can infer that the treatment variable causes the effect measured by the outcome variable. The term *effect* is sometimes reserved to the causal association that may be established in experimental studies, but is often used more loosely to mean any association between predictors and outcomes. This loose usage may mask the very different nature of the associations detected in these two types of studies.

When we use models for prediction only, the caveats about inference of the first application of linear models do not apply. It does not matter much that our predictors may be confounded, as what counts here is just a good prediction of the outcome. We are not overly concerned here if the data were produced in an experimental or an observational study.

---
summary: Linear models are computational tools that process numerical data to produce outputs, but they do not interpret the data as representing aspects of reality; they only see numbers. Their primary function is to relate input numbers to measurements and observations in the real world, interpreting the output based on this knowledge. Linear models optimize the use of data to enhance their predictive accuracy, but their interpretation remains limited to numerical associations rather than real-world understanding.
~txthash: mx5ZQEPsMbLIf8ETTXguFw
titles: Chapter 1 - What are linear models? - A third way of looking at linear models
source: Ch1
questions: "How do linear models function as computational devices that process data? - Why is it important to relate the numerical inputs of a linear model to real-world measurements? - How can linear models quantify the influence of predictors on outcomes?"
---
### A third way of looking at linear models

We may perhaps add that there is a third way to look at linear models, which is as computational devices that crunch numbers to output sophisticated averages of the data. From this perspective, the linear model does not care at all what kind of numbers are fed to it, as it will usually be able to compute something from them. Linear models are not oracles that can divine aspects of reality: all that they see is just numbers. Therefore, an important task when estimating linear models is our capacity to relate these numbers to measurements and observations in the real world, and interpret the output of the model in the light of this knowledge. The distinction between the interpretation of the associations in observational and experimental studies is one example of this task. Nevertheless, linear models can quantify the relative role of predictors on outcomes in ways that are not available to simple inspection. Furthermore, they can make optimal use of the information contained in the data, unlike a human observer.
"""

from lmm_education.config.config import EncodingModel


def setUpModule():
    # crate a ragged markdown on disk
    from lmm.markdown.parse_markdown import (
        parse_markdown_text,
        blocklist_haserrors,
        save_blocks,
    )
    from lmm.utils.logging import LoglistLogger

    logger = LoglistLogger()

    blocks = parse_markdown_text(document)
    if blocklist_haserrors(blocks):
        raise RuntimeError("Uexpected errors in markdown")

    save_blocks("TestRaggedDocument.md", blocks, logger)


def tearDownModule():
    # delete TestRaggedDocument.md
    import os

    if os.path.exists("TestRaggedDocument.md"):
        os.remove("TestRaggedDocument.md")

    # delete folder ./test_storage, it is exists
    if os.path.exists("./test_storage"):
        import shutil

        try:
            shutil.rmtree("./test_storage")
        except Exception:
            pass


def get_text_block_count() -> int:
    from lmm.markdown.parse_markdown import (
        parse_markdown_text,
        TextBlock,
    )
    from lmm.scan.scan_split import scan_split

    blocks = parse_markdown_text(document)
    blocks = scan_split(blocks)
    return len([b for b in blocks if isinstance(b, TextBlock)])


def get_chunks_count() -> int:
    from lmm.markdown.parse_markdown import (
        parse_markdown_text,
        MetadataBlock,
    )
    from lmm.scan.scan_keys import SUMMARIES_KEY
    from lmm_education.config.config import ConfigSettings

    settings = ConfigSettings()
    count_chunks = get_text_block_count()

    if settings.RAG.summaries:
        blocks = parse_markdown_text(document)
        meta_blocks = [
            b for b in blocks if isinstance(b, MetadataBlock)
        ]
        count_chunks += len(
            [m for m in meta_blocks if m.get_key(SUMMARIES_KEY)]
        )

    return count_chunks


class TestIngestionRetrieval(unittest.TestCase):

    def test_integration_retrieval_content(self):
        pass
        from lmm_education.ingest import markdown_upload
        from lmm_education.config.config import ConfigSettings

        # 'dense_model': "SentenceTransformers/distiluse-base-multilingual-cased-v1",
        config = ConfigSettings(
            embeddings={
                'dense_model': "SentenceTransformers/all-MiniLM-L6-v2",
                'sparse_model': "Qdrant/bm25",
            },
            storage={'folder': "./test_storage"},
            database={
                'collection_name': "TestContent",
                'companion_collection': None,  # "Documents",
            },
            RAG={
                'questions': True,
                'annotation_model': {'own_properties': ['questions']},
                'encoding_model': "content",  # "multivector",
            },
            textSplitter={"splitter": "default", "threshold": 125},
        )
        idss = markdown_upload(
            "TestRaggedDocument.md",
            config_opts=config,
        )
        self.assertTrue(len(idss) > 0)
        self.assertEqual(len(idss), get_chunks_count())

        from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
            QdrantVectorStoreRetriever as Qdr,
        )

        retriever = Qdr.from_config_settings(config)

        docs = retriever.invoke("What are observational studies?")
        retriever.client.close()
        self.assertTrue(len(docs) > 0)

    def _do_test_integration_retrieval(
        self, encoding_model: EncodingModel
    ):
        from lmm_education.ingest import markdown_upload
        from lmm_education.config.config import ConfigSettings

        # 'dense_model': "SentenceTransformers/distiluse-base-multilingual-cased-v1",
        # 'dense_model': "SentenceTransformers/all-MiniLM-L6-v2",
        config = ConfigSettings(
            embeddings={
                'dense_model': "SentenceTransformers/all-MiniLM-L6-v2",
                'sparse_model': "Qdrant/bm25",
            },
            storage={'folder': "./test_storage"},
            database={
                'collection_name': "Test" + str(encoding_model.value),
                'companion_collection': None,  # "Documents",
            },
            RAG={
                'questions': True,
                'annotation_model': {'own_properties': ['questions']},
                'encoding_model': encoding_model,
            },
        )
        idss = markdown_upload(
            "TestRaggedDocument.md",
            config_opts=config,
        )
        self.assertTrue(len(idss) > 0)
        self.assertEqual(len(idss), get_chunks_count())

        from lmm_education.stores.langchain.vector_store_qdrant_langchain import (
            QdrantVectorStoreRetriever as Qdr,
        )

        retriever = Qdr.from_config_settings(config)

        chunks = retriever.invoke("What are observational studies?")
        retriever.client.close()
        self.assertTrue(len(chunks) > 0)

    def test_integration_retrieval_models(self):
        # CONTENT already tested, test rest
        models: list[EncodingModel] = [
            EncodingModel.SPARSE,
            EncodingModel.SPARSE_CONTENT,
            EncodingModel.SPARSE_MERGED,
            EncodingModel.SPARSE_MULTIVECTOR,
            EncodingModel.MULTIVECTOR,
            EncodingModel.MERGED,
        ]
        for m in models:
            self._do_test_integration_retrieval(m)


if __name__ == "__main__":
    unittest.main()
