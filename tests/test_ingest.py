import unittest
import io

from lmm.markdown.parse_markdown import parse_markdown_text
from lmm.utils.logging import LoglistLogger

# from lmm.scan.scan_keys import TITLES_KEY, QUESTIONS_KEY, SUMMARY_KEY

from lmm_education.ingest import (
    blocklist_encode,
    markdown_upload,
    ConfigSettings,
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
        output = io.StringIO()
        logger = LoglistLogger()
        ids = markdown_upload(
            "Chapter_1.Rmd",
            config_opts=opts,
            save_files=output,
            ingest=False,
            logger=logger,
        )
        print("\n".join(logger.get_logs()))
        self.assertTrue(logger.count_logs(level=1) == 0)
        self.assertTrue(bool(ids))


if __name__ == "__main__":
    unittest.main()
