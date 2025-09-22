import unittest
import io

from qdrant_client import QdrantClient

from lmm.markdown.parse_markdown import parse_markdown_text
from lmm.utils.logging import LoglistLogger

# from lmm.scan.scan_keys import TITLES_KEY, QUESTIONS_KEY, SUMMARY_KEY

from lmm_education.ingest import (
    blocklist_encode,
    markdown_upload,
    ConfigSettings,
    initialize_client,
)
from lmm_education.stores import EncodingModel


document = """
---
author: Roberto Viviani
date: '2025-05-04'
docid: Ch1
keywords:
- linear model
- observational study
- experimental study
title: Chapter 1
---

## Introduction

These lectures have the purpose of having a second look at linear models from a practical perspective. We will assume that basic knowledge of statistics (t tests, significance levels, etc.) is already present, as it may have been given in a first introductory course dealing with notions such as testing and statistical significance.

In the whole course we will use R to specify and fit the models. We will assume basic familiarity with R commands, but there will be no need to write programs in R. While initially R might seem an additional hurdle, its use is invaluable because it exposes the logic of linear models directly. It is also the language used by professional statisticians to make available new techniques to the world. If you know R, chances are that you can find a way to apply the most state-of-the-art statistical approach to your data, should you one day need it.

## What are linear models?

Linear models and their generalizations constitute the majority of the statistical models used in practice. Here, we will look at linear models from a practical perspective, emphasizing the issues in correctly applying them and in understanding their output.

Linear models capture the association between a set of variables, the _predictors_ (also known as _independent variables_), and an _outcome_ variable (or _dependent variable_). In the simplest setting, this association may be approximated and displayed by a line relating a predictor and the outcome. However, the generalizations of linear model allow capturing much more complex associations, as we shall see. 

There are two broad ways of using linear models. In the first, which is perhaps the most common, we are interested in assessing the relationship between predictors and the outcome in order to establish if this relationship is "significant". 

The second use of linear models is to predict the outcome given certain values of the predictors. This use of linear models is the same as in machine learning.[^1] In this case, after the fit of a linear model has been computed, one may use values of the predictors that the model had not seen to predict the outcome.

### Observational and experimental studies

We will first discuss issues arising from assessing the significance of associations.

When we look at the significance of the association between predictors and outcome, it is important to distinguish between two different settings. In the first, we have variables that we have observed in the field. For example, early traumas may exert an influence on the predisposition to psychopathology in adult age. We do not have any way to change the occurrence of traumas in the past and we therefore look at their consequences in a sample from the population. Studies of this type are called _observational_. An important issue in observational studies is that predictors may be _confounded_ by other factors affecting the outcome. For example, it is conceivable that traumas may occur more frequently in adverse socioeconomic conditions, and these latter may in turn adversely affect the predisposition to psychopathology. When we assess the association of traumas and psychopathology, it may include the effects of socioeconomic conditions, as traumatized individuals are also those that are most disadvantaged socioeconomically.

In the second setting, the predictor of interest is a variable representing an experimental manipulation, such as treatment, changes in aspects of a cognitive task, etc. A key aspect of _experimental_ studies is that the value of the predictor of interest, being determined by you, the experimenter, can be randomized. For this reason, at least in the long term, the action of other possible variables on the outcome cancels out, as there is no relationship between these potential confounders and the predictor of interest. We cannot randomize traumas, but if we could, we would no longer have the potential confounder of adverse socioeconomic conditions, because randomization ensures that, in the long run, individuals from all conditions are equally traumatized.

Importantly, the way in which we estimate models in observational and experimental studies is exactly the same. However, the conclusions that we may draw from establishing the likely existence of an association between predictor and outcome differ in these two cases. In observational studies, we can only infer the mere association, unless a sometimes fairly extensive efforts are made on being able to contain the effect of confounders. In experimental studies, we can infer that the treatment variable causes the effect measured by the outcome variable. The term _effect_ is sometimes reserved to the causal association that may be established in experimental studies, but is often used more loosely to mean any association between predictors and outcomes. This loose usage may mask the very different nature of the associations detected in these two types of studies.

When we use models for prediction only, the caveats about inference of the first application of linear models do not apply. It does not matter much that our predictors may be confounded, as what counts here is just a good prediction of the outcome. We are not overly concerned here if the data were produced in an experimental or an observational study.

"""

blocklist = parse_markdown_text(document)

import lmm.scan.scan_keys as keys

CHUNK_KEY = '~chunk'


class TestIngest(unittest.TestCase):

    def test_settings_minimal(self):
        listlogger = LoglistLogger()
        opts = ConfigSettings(
            storage=":memory:",
            encoding_model=EncodingModel.CONTENT,
            questions=False,
            summaries=False,
            companion_collection=None,
            text_splitter={'splitter': "default", 'threshold': 75},
        )
        chunks = blocklist_encode(blocklist, opts, listlogger)
        self.assertTrue(len(chunks[0]) > 0)
        self.assertTrue(len(chunks[1]) == 0)
        for ch in chunks[0]:
            self.assertTrue(ch.uuid)
            self.assertTrue(keys.TITLES_KEY in ch.metadata)
            self.assertTrue(bool(ch.metadata[keys.TITLES_KEY]))
            self.assertFalse(bool(ch.sparse_encoding))

    def test_settings_sparsetitles(self):
        listlogger = LoglistLogger()
        opts = ConfigSettings(
            storage=":memory:",
            encoding_model=EncodingModel.SPARSE_CONTENT,
            questions=False,
            summaries=False,
            companion_collection=None,
            text_splitter={'splitter': "default", 'threshold': 75},
        )
        chunks = blocklist_encode(blocklist, opts, listlogger)
        self.assertTrue(len(chunks[0]) > 0)
        self.assertTrue(len(chunks[1]) == 0)
        for ch in chunks[0]:
            self.assertTrue(ch.uuid)
            self.assertTrue(bool(ch.metadata[keys.TITLES_KEY]))
            self.assertEqual(
                ch.sparse_encoding, ch.metadata[keys.TITLES_KEY]
            )

    def test_settings_densetitles(self):
        listlogger = LoglistLogger()
        opts = ConfigSettings(
            storage=":memory:",
            encoding_model=EncodingModel.MERGED,
            questions=False,
            summaries=False,
            companion_collection=None,
            text_splitter={'splitter': "default", 'threshold': 75},
        )
        chunks = blocklist_encode(blocklist, opts, listlogger)
        self.assertTrue(len(chunks[0]) > 0)
        self.assertTrue(len(chunks[1]) == 0)
        for ch in chunks[0]:
            self.assertTrue(ch.uuid)
            self.assertTrue(bool(ch.metadata[keys.TITLES_KEY]))
            self.assertIn(
                ch.metadata[keys.TITLES_KEY], ch.dense_encoding
            )

    def test_settings_sparsequestions(self):
        listlogger = LoglistLogger()
        opts = ConfigSettings(
            storage=":memory:",
            encoding_model=EncodingModel.SPARSE_CONTENT,
            questions=True,
            summaries=False,
            companion_collection=None,
            text_splitter={'splitter': "default", 'threshold': 75},
        )
        chunks = blocklist_encode(blocklist, opts, listlogger)
        self.assertTrue(len(chunks[0]) > 0)
        self.assertTrue(len(chunks[1]) == 0)
        for ch in chunks[0]:
            self.assertTrue(ch.uuid)
            self.assertTrue(bool(ch.metadata[keys.QUESTIONS_KEY]))
            self.assertIn(
                ch.metadata[keys.TITLES_KEY], ch.sparse_encoding
            )
            self.assertIn(
                ch.metadata[keys.QUESTIONS_KEY], ch.sparse_encoding
            )

    def test_settings_sparsedensequestions(self):
        listlogger = LoglistLogger()
        opts = ConfigSettings(
            storage=":memory:",
            encoding_model=EncodingModel.SPARSE_MERGED,
            questions=True,
            summaries=False,
            companion_collection=None,
            text_splitter={'splitter': "default", 'threshold': 75},
        )
        chunks = blocklist_encode(blocklist, opts, listlogger)
        self.assertTrue(len(chunks[0]) > 0)
        self.assertTrue(len(chunks[1]) == 0)
        for ch in chunks[0]:
            self.assertTrue(ch.uuid)
            self.assertTrue(bool(ch.metadata[keys.QUESTIONS_KEY]))
            self.assertIn(
                ch.metadata[keys.TITLES_KEY], ch.sparse_encoding
            )
            self.assertIn(
                ch.metadata[keys.QUESTIONS_KEY], ch.sparse_encoding
            )
            self.assertIn(
                ch.metadata[keys.QUESTIONS_KEY], ch.dense_encoding
            )

    def test_settings_summaries(self):
        listlogger = LoglistLogger()
        opts = ConfigSettings(
            storage=":memory:",
            encoding_model=EncodingModel.CONTENT,
            questions=False,
            summaries=True,
            companion_collection=None,
            text_splitter={'splitter': "default", 'threshold': 75},
        )
        chunks = blocklist_encode(blocklist, opts, listlogger)
        self.assertTrue(len(chunks[0]) > 0)
        self.assertTrue(len(chunks[1]) == 0)
        counter: int = 0
        for ch in chunks[0]:
            self.assertTrue(ch.uuid)
            if 'type' in ch.metadata:
                counter += 1
        self.assertEqual(counter, 3)

    def test_settings_companion(self):
        listlogger = LoglistLogger()
        opts = ConfigSettings(
            storage=":memory:",
            encoding_model=EncodingModel.CONTENT,
            questions=False,
            summaries=False,
            companion_collection="documents",
            text_splitter={'splitter': "default", 'threshold': 75},
        )
        chunks = blocklist_encode(blocklist, opts, listlogger)
        self.assertTrue(len(chunks[0]) > 0)
        self.assertTrue(len(chunks[1]) > 0)
        self.assertEqual(len(chunks[1]), 3)
        for ch in chunks[1]:
            self.assertTrue(ch.uuid)
            self.assertTrue(bool(ch.metadata[keys.TITLES_KEY]))


class TestLoadMarkdown(unittest.TestCase):

    @staticmethod
    def _create_config_settings() -> ConfigSettings:
        return ConfigSettings(
            storage=":memory:",
            encoding_model=EncodingModel.CONTENT,
            questions=False,
            summaries=False,
            companion_collection=None,
            text_splitter={'splitter': "default", 'threshold': 75},
        )

    def test_load_nonexistent_markdown(self):
        opts = self._create_config_settings()
        logger = LoglistLogger()
        ids = markdown_upload(
            ["This.md", "That.md"],
            config_opts=opts,
            save_files=False,
            ingest=False,
            logger=logger,
        )
        print("\n".join(logger.get_logs()))
        self.assertTrue(logger.count_logs(level=1) > 0)
        self.assertFalse(bool(ids))

    def test_load_markdown(self):
        opts = self._create_config_settings()
        logger = LoglistLogger()
        client: QdrantClient | None = initialize_client(opts, logger)
        if client is None:
            print("\n".join(logger.get_logs(level=0)))
            raise Exception("Could not initialize client")
        self.assertTrue(logger.count_logs(level=1) == 0)
        ids = markdown_upload(
            "Chapter_1.Rmd",
            config_opts=opts,
            save_files=False,
            ingest=True,
            client=client,
            logger=logger,
        )
        print("\n".join(logger.get_logs(level=0)))
        self.assertTrue(logger.count_logs(level=1) == 0)
        self.assertTrue(len(ids) > 0)

        # check using qdrant API that records are there
        from qdrant_client.models import Record

        id = ids[0][0]
        records: list[Record] = client.retrieve(
            collection_name=opts.collection_name,
            ids=[id],
            with_payload=True,
        )
        self.assertTrue(len(records) > 0)

    def test_load_markdown_companion(self):
        opts = ConfigSettings(
            storage=":memory:",
            encoding_model=EncodingModel.CONTENT,
            questions=False,
            summaries=False,
            companion_collection="documents",
            text_splitter={'splitter': "default", 'threshold': 75},
        )

        logger = LoglistLogger()
        client: QdrantClient | None = initialize_client(opts, logger)
        if client is None:
            print("\n".join(logger.get_logs(level=0)))
            raise Exception("Could not initialize client")
        self.assertTrue(logger.count_logs(level=1) == 0)
        ids = markdown_upload(
            "Chapter_1.Rmd",
            config_opts=opts,
            save_files=False,
            ingest=True,
            client=client,
            logger=logger,
        )
        print("\n".join(logger.get_logs(level=0)))
        self.assertTrue(logger.count_logs(level=1) == 0)
        self.assertTrue(len(ids) > 0)

        # check using qdrant API that records are there
        from qdrant_client.models import Record

        id = ids[0][1]
        records: list[Record] = client.retrieve(
            collection_name=opts.companion_collection,
            ids=[id],
            with_payload=True,
        )
        self.assertTrue(len(records) > 0)


class TestMarkdownQueries(unittest.TestCase):

    @staticmethod
    def _create_config_settings(
        companion: bool = False,
    ) -> ConfigSettings:
        return ConfigSettings(
            storage=":memory:",
            encoding_model=EncodingModel.CONTENT,
            questions=False,
            summaries=False,
            companion_collection="documents" if companion else None,
            text_splitter={'splitter': "default", 'threshold': 75},
        )

    def test_ingest_markdown(self):
        opts = self._create_config_settings()
        output = io.StringIO()
        logger = LoglistLogger()
        client: QdrantClient | None = initialize_client(opts, logger)
        if not client:
            print("\n".join(logger.get_logs(level=1)))
            Exception("Database could not be initialized.")
            return
        self.assertTrue(logger.count_logs(level=1) == 0)
        ids = markdown_upload(
            "Chapter_1.Rmd",
            config_opts=opts,
            save_files=output,
            ingest=True,
            client=client,
            logger=logger,
        )
        self.assertTrue(logger.count_logs(level=1) == 0)
        self.assertTrue(len(ids) > 0)
        self.assertTrue(len(output.getvalue()) > 0)

        from lmm_education.stores import query
        from lmm_education.stores.vector_store_qdrant import (
            encoding_to_qdrantembedding_model,
        )

        results = query(
            client,
            collection_name=opts.collection_name,
            model=encoding_to_qdrantembedding_model(
                opts.encoding_model
            ),
            querytext="What are the main uses of linear models?",
            limit=4,
            payload=['page_content'],
            logger=logger,
        )
        print("\n".join(logger.get_logs(level=0)))
        self.assertTrue(logger.count_logs(level=1) == 0)
        self.assertTrue(len(results) > 0)

    def test_ingest_markdown_langchain(self):
        opts = self._create_config_settings()
        output = io.StringIO()
        logger = LoglistLogger()
        client: QdrantClient | None = initialize_client(opts, logger)
        if not client:
            print("\n".join(logger.get_logs(level=1)))
            Exception("Database could not be initialized.")
            return
        self.assertTrue(logger.count_logs(level=1) == 0)
        ids = markdown_upload(
            "Chapter_1.Rmd",
            config_opts=opts,
            save_files=output,
            ingest=True,
            client=client,
            logger=logger,
        )
        self.assertTrue(logger.count_logs(level=1) == 0)
        self.assertTrue(len(ids) > 0)
        self.assertTrue(len(output.getvalue()) > 0)

        from lmm_education.stores.langchain import (
            QdrantVectorStoreRetriever,
        )
        from lmm_education.stores.vector_store_qdrant import (
            encoding_to_qdrantembedding_model,
        )

        retriever = QdrantVectorStoreRetriever(
            client,
            opts.collection_name,
            encoding_to_qdrantembedding_model(opts.encoding_model),
        )

        # this how to call our function directly
        # from langchain_core.callbacks.manager import CallbackManager

        # results = retriever._get_relevant_documents(
        #     "What are the main uses of linear models?",
        #     run_manager=CallbackManager.configure().on_retriever_start(
        #         None, ""
        #     ),
        # )
        results = retriever.invoke(
            "What are the main uses of linear models?"
        )
        print("\n".join(logger.get_logs(level=0)))
        self.assertTrue(logger.count_logs(level=1) == 0)
        self.assertTrue(len(results) > 0)

    def xtest_load_ingest_markdown_langchain2(self):
        # does not work if opts are :memory:
        opts = ConfigSettings(
            storage=":memory:",
            encoding_model=EncodingModel.CONTENT,
            questions=True,
            summaries=False,
            companion_collection=None,
            text_splitter={'splitter': "default", 'threshold': 75},
        )

        output = io.StringIO()
        logger = LoglistLogger()
        client: QdrantClient | None = initialize_client(opts, logger)
        if not client:
            print("\n".join(logger.get_logs(level=1)))
            Exception("Database could not be initialized.")
            return
        self.assertTrue(logger.count_logs(level=1) == 0)
        ids = markdown_upload(
            "Chapter_1.Rmd",
            config_opts=opts,
            save_files=output,
            ingest=True,
            client=client,
            logger=logger,
        )
        self.assertTrue(logger.count_logs(level=1) == 0)
        self.assertTrue(len(ids) > 0)
        self.assertTrue(len(output.getvalue()) > 0)

        from lmm_education.stores.langchain import (
            QdrantVectorStoreRetriever as QdrantRetriever,
        )

        retriever = QdrantRetriever.from_config_settings(opts)
        results = retriever.invoke(
            "What are the main uses of linear models?"
        )
        print("\n".join(logger.get_logs(level=0)))
        self.assertTrue(logger.count_logs(level=1) == 0)
        self.assertTrue(len(results) > 0)

    def test_ingest_markdown_grouped(self):
        from lmm.scan.scan_keys import GROUP_UUID_KEY

        opts = self._create_config_settings(True)
        output = io.StringIO()
        logger = LoglistLogger()
        client: QdrantClient | None = initialize_client(opts, logger)
        if not client:
            print("\n".join(logger.get_logs(level=1)))
            Exception("Database could not be initialized.")
            return
        self.assertTrue(logger.count_logs(level=1) == 0)
        ids = markdown_upload(
            "Chapter_1.Rmd",
            config_opts=opts,
            save_files=output,
            ingest=True,
            client=client,
            logger=logger,
        )
        self.assertTrue(logger.count_logs(level=1) == 0)
        self.assertTrue(len(ids) > 0)
        self.assertTrue(len(output.getvalue()) > 0)

        from lmm_education.stores import (
            query_grouped,
            groups_to_points,
            points_to_text,
        )
        from lmm_education.stores.vector_store_qdrant import (
            encoding_to_qdrantembedding_model,
            ScoredPoint,
        )

        results = query_grouped(
            client,
            opts.collection_name,
            opts.companion_collection,
            encoding_to_qdrantembedding_model(opts.encoding_model),
            "What are the main uses of linear models?",
            limit=1,
            group_field=GROUP_UUID_KEY,
            logger=logger,
        )
        print("\n".join(logger.get_logs(level=0)))
        self.assertTrue(logger.count_logs(level=1) == 0)

        result_points: list[ScoredPoint] = groups_to_points(results)
        result_text: list[str] = points_to_text(result_points)

        self.assertLess(0, len(result_points))
        companion_uuids = set([id[1] for id in ids])
        self.assertTrue(results.groups[0].id in companion_uuids)
        self.assertTrue(len(result_text) > 0)
        print(len(result_text))


if __name__ == "__main__":
    unittest.main()
