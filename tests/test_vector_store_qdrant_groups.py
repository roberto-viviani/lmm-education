"""Tests for grouped qdrant queries"""

import unittest

from langchain_text_splitters import RecursiveCharacterTextSplitter

from lmm.markdown.parse_markdown import (
    Block,
    MetadataBlock,
    parse_markdown_text,
    blocklist_haserrors,
)
from lmm.scan.scan_keys import GROUP_UUID_KEY, UUID_KEY
from lmm.scan.scan_rag import ScanOpts, scan_rag
from lmm.scan.scan_split import scan_split
from lmm_education.stores.vector_store_qdrant import (
    QdrantClient,
    Point,
    ScoredPoint,
    GroupsResult,
    QdrantEmbeddingModel,
    EncodingModel,
    points_to_text,
    points_to_ids,
    groups_to_points,
    initialize_collection,
    encoding_to_qdrantembedding_model,
    upload,
    query,
    query_grouped,
)
from lmm_education.stores.chunks import (
    Chunk,
    blocks_to_chunks,
)

# A global client object (for now)
QDRANT_SOURCE = ":memory:"
COLLECTION_MAIN = "Main"
COLLECTION_DOCS = "Main_docs"
client = QdrantClient(QDRANT_SOURCE)

document: str = """
---
author: Roberto Viviani
date: '2025-05-04'
docid: Ch1
keywords:
- linear model
- observational study
- experimental study
output:
  html_document: default
  md_document: default
  word_document: default
title: Chapter 1
---

---
questions: What is the purpose of the course?
---
## Introduction

These lectures have the purpose of having a second look at linear models from a practical perspective. We will assume that basic knowledge of statistics (t tests, significance levels, etc.) is already present, as it may have been given in a first introductory course dealing with notions such as testing and statistical significance.

In the whole course we will use R to specify and fit the models. We will assume basic familiarity with R commands, but there will be no need to write programs in R. While initially R might seem an additional hurdle, its use is invaluable because it exposes the logic of linear models directly. It is also the language used by professional statisticians to make available new techniques to the world. If you know R, chances are that you can find a way to apply the most state-of-the-art statistical approach to your data, should you one day need it.

## What are linear models?

Linear models and their generalizations constitute the majority of the statistical models used in practice. Here, we will look at linear models from a practical perspective, emphasizing the issues in correctly applying them and in understanding their output.

Linear models capture the association between a set of variables, the _predictors_ (also known as _independent variables_), and an _outcome_ variable (or _dependent variable_). In the simplest setting, this association may be approximated and displayed by a line relating a predictor and the outcome. However, the generalizations of linear model allow capturing much more complex associations, as we shall see. 

There are two broad ways of using linear models. In the first, which is perhaps the most common, we are interested in assessing the relationship between predictors and the outcome in order to establish if this relationship is "significant". 

The second use of linear models is to predict the outcome given certain values of the predictors. This use of linear models is the same as in machine learning.[^1] In this case, after the fit of a linear model has been computed, one may use values of the predictors that the model had not seen to predict the outcome.

[^1]: Machine learning is the field of artificial intelligence that is concerned with training programs to accomplish tasks based on a training set. 

---
summary:  The text discusses the importance of distinguishing between observational and experimental studies when assessing the significance of associations between predictors and outcomes. In observational studies, predictors may be confounded by other factors, making it difficult to infer causality, while experimental studies allow for randomization, which can help to control for confounders and infer causal effects. The methods used to estimate models in both types of studies are the same, but the conclusions drawn from them differ significantly. Additionally, when using models purely for prediction, the concerns about confounding and inference are less critical, as the goal is simply to accurately predict the outcome.'
---
### Observational and experimental studies

We will first discuss issues arising from assessing the significance of associations.

When we look at the significance of the association between predictors and outcome, it is important to distinguish between two different settings. In the first, we have variables that we have observed in the field. For example, early traumas may exert an influence on the predisposition to psychopathology in adult age. We do not have any way to change the occurrence of traumas in the past and we therefore look at their consequences in a sample from the population. Studies of this type are called _observational_. An important issue in observational studies is that predictors may be _confounded_ by other factors affecting the outcome. For example, it is conceivable that traumas may occur more frequently in adverse socioeconomic conditions, and these latter may in turn adversely affect the predisposition to psychopathology. When we assess the association of traumas and psychopathology, it may include the effects of socioeconomic conditions, as traumatized individuals are also those that are most disadvantaged socioeconomically.

In the second setting, the predictor of interest is a variable representing an experimental manipulation, such as treatment, changes in aspects of a cognitive task, etc. A key aspect of _experimental_ studies is that the value of the predictor of interest, being determined by you, the experimenter, can be randomized. For this reason, at least in the long term, the action of other possible variables on the outcome cancels out, as there is no relationship between these potential confounders and the predictor of interest. We cannot randomize traumas, but if we could, we would no longer have the potential confounder of adverse socioeconomic conditions, because randomization ensures that, in the long run, individuals from all conditions are equally traumatized.

Importantly, the way in which we estimate models in observational and experimental studies is exactly the same. However, the conclusions that we may draw from establishing the likely existence of an association between predictor and outcome differ in these two cases. In observational studies, we can only infer the mere association, unless a sometimes fairly extensive efforts are made on being able to contain the effect of confounders. In experimental studies, we can infer that the treatment variable causes the effect measured by the outcome variable. The term _effect_ is sometimes reserved to the causal association that may be established in experimental studies, but is often used more loosely to mean any association between predictors and outcomes. This loose usage may mask the very different nature of the associations detected in these two types of studies.

When we use models for prediction only, the caveats about inference of the first application of linear models do not apply. It does not matter much that our predictors may be confounded, as what counts here is just a good prediction of the outcome. We are not overly concerned here if the data were produced in an experimental or an observational study.

---
summary: This section introduces the model equation, emphasizing its importance in understanding the linear model.
...
## The model equation

It is now time to see a linear model in action. Let us assume we measured depressive symptoms (the outcome) with a self-rating scale and we are interested in exploring their association with sex. We may formulate the model as follows:

depression = baseline + female + error

Here, $female$ encodes sex by taking the value 1 for women and 0 for men. The outcome we measured (depressive symptoms) is decomposed by the model into the sum of three terms: the baseline depressiveness, i.e. the average depressive symptoms of males; the difference in depressive symptoms observed on average in females relative to males, and a final term accounting for the difference in depressive symptoms measured in each individuals relative to these averages. This expression applies for each observation. If we index an observation with $i$, we may write

$$depression_i = baseline + female_i * female_coefficient + error_i$$

This equation is called the _model equation_. At this point, you might be tempted to skip over the model equation and think that you may deal with it later or avoid using equations altogether. Don't! The model equation is crucial to understand linear models. Once mastered, it becomes your most helpful aid in understanding the model. It is therefore crucial to become familiar with it. 
This model equation means that in individual $i$, the measured depressive symptoms $depression_i$ are the sum of  $baseline$, which is the average depressive symptoms in male individuals, of the average difference in depressive symptoms in females $female_coefficient$, and of the errors $error_i$. The $female_coefficient$ is applied to all all data, but here in males it has no influence as it is multiplied by $female$, which is zero in males. In females, the $female_coefficient$ is multiplied by one and remains equal to itself. The errors are the difference between the observed depressiveness in each individual and these estimated averages. Note that there is one baseline and one female coefficient for the whole dataset. These are the _coefficients_ of the model. The baseline coefficient is sometimes called _constant term_ or _intercept_.

When the model is _fitted_, we obtain the estimates of the coefficients. To do this, we use the following line in the R console:

```{r}
fit <- lm(depression ~ female, data = depr_sample)
```

where depr_sample is the data frame containing the columns `depression` and `female`. The expression `depression ~ female` is the model equation, except for the baseline term, which is added automatically by R. We could also have written this term explicitly in the model equation as follows,

```{r}
fit <- lm(depression ~ 1 + female, data = depr_sample)
```

obtaining the same model. We can now inspect the coefficients of the model fit:

```{r}
summary(fit)
```

To understand what these coefficients mean, let us put them in the model equation. We obtain

$$depression_i = 24 + female_i * 4 + error_i$$

To calculate the predicted depressiveness, we set the value of the female predictor to one or zero, depending on the sex of the individual we are predicting. It follows that 24 is the predicted depressive symptom score in males, and 4 is the predicted difference in these scores from males to females. It also follows that the predicted depression score in females is 24 + 4 = 28 (in many samples of self-reported depressiveness, one often finds higher scores in females, although this finding may fail to be confirmed by other measures of depressiveness).

While this is a linear model, it is also a two-sample _t_ test. In a two-sample _t_ test we test the difference between means. Here, these means are 24 and 28. The linear model is a very general formulation, relative to which other models are special cases. 

Another case of a linear model that you might not have suspected to be one is the mean:

$$depression_i = constant_term + error_i$$

where we used the name _constant term_ for the baseline, estimated as the average of all observed depression scores in the whole sample (which will differ from the estimate of the male average of the previous model). This is the simplest possible linear model, and we will come back to it later to help intuition about the properties of the fitted model.

## Regression on a quantitative variable

Probably, what comes to mind first when hearing about a regression is a linear model of a quantitative variable. Here, we assessed traumas in childhood using the well-known child trauma questionnaire (CTQ). We are interested in assessing the association between child traumas and depressiveness. Hence, our model equation is

$$depression_i = baseline + CTQ_i * CTQ_coefficient + error_i$$

We fit this model by using the model equation in R,

```{r}
fit <- lm(depression ~ CTQ, data = depr_sample)
```

If we put the coefficients back in the model equation, we obtain

$$depression_i = 25 + CTQ_i * 0.13 + error_i$$

Here, 25 is our estimated baseline depressiveness, i.e. the predicted depressiveness in individuals with a CTQ score of zero. You can see that this is the case because, if $CTQ_i$ is zero and this value is substituted in the equation, we obtain

$$depression_i = 25 + error_i$$

which says that in this case the predicted depressiveness is 25. 

The model equation maintains the same meaning here and in the previous example. The estimated coefficient of CTQ, 0.13, is now the predicted difference in depressiveness in individuals with CTQ score higher by one point. In individuals where CTQ is 1, the predicted depressiveness is 25 + 1 * 0.13 = 25.13. In individuals with a CTQ score of 10, the predicted depressiveness is 25 + 10 * 0.13 = 26.3.

This is an observational study, so we cannot say that CTQ _causes_ depression in adulthood from these data. For this reason, some would also frown on you if saying that _the effect of_ CTQ on depressiveness is 0.13. Similar misunderstandings might arise if one says that an increase of one unit in CTQ _increases_ depressiveness scores by 0.13. The best way to understand this coefficient is that, in the sampled individuals whose CTQ score differed by one unit the average difference in depressiveness was 0.13. This is now also our best guess about the average difference in depressiveness scores in the population between individuals whose CTQ scores differ by one unit. This difference is the same across all levels of CTQ -- this is a consequence of estimating the association between CTQ and depressiveness as linear.

## Multiple regression

Sex and traumas can be used in the same model by including both in the model equation.

```{r}
fit <- lm(depression ~ female + CTQ, data = depr_sample)
summary(fit)
```

The fitted model equation is here

$$depression_i = 25 + female_i * 3.6 + CTQ_i * 0.12 + error_i$$

which may be read as follows. 25 is the expected baseline depressiveness in males with a CTQ score of zero. In females, the predicted depressiveness increases by 3.6. For each additional unit score in the CTQ, the predicted depressiveness increases by 0.12.

Note that the estimated coefficients for sex and CTQ differ from those of the previous two models, where they were estimated in isolation in separate models. This is what always happens, unless `female` and `CTQ` are unrelated to each other. Suppose, for example, that you sampled males and females such that the average CTQ score in these two groups is exactly the same. In this case, the two predictors are _orthogonal_, and the coefficients obtained in the multiple regression and in the separate regressions will be the same. This happens practically never in observational studies, unless sampling techniques are used.

A crucial consequence of putting predictors together in the model is that the coefficients estimates are _adjusted_ for each other.

"""

# prepare blocklist for RAG
blocklist: list[Block] = parse_markdown_text(document)
if blocklist_haserrors(blocklist):
    raise Exception("Invalid markdown")


class TestEncoding(unittest.TestCase):

    def test_encode(self):
        # init collections
        encoding_model_main: EncodingModel = EncodingModel.CONTENT
        embedding_model_main: QdrantEmbeddingModel = (
            encoding_to_qdrantembedding_model(encoding_model_main)
        )
        encoding_model_companion: EncodingModel = (
            EncodingModel.CONTENT
        )
        embedding_model_companion: QdrantEmbeddingModel = (
            encoding_to_qdrantembedding_model(encoding_model_main)
        )

        if not initialize_collection(
            client, COLLECTION_MAIN, embedding_model_main
        ):
            raise Exception("Could not initialize main collection")
        if not initialize_collection(
            client, COLLECTION_DOCS, embedding_model_companion
        ):
            raise Exception(
                "Could not initialize companion collection"
            )

        blocks: list[Block] = scan_rag(
            blocklist,
            ScanOpts(titles=True, textid=True, UUID=True),
        )

        # ingest text into companion collection
        companion_chunks: list[Chunk] = blocks_to_chunks(
            blocks, encoding_model_companion
        )
        companion_points: list[Point] = upload(
            client,
            COLLECTION_DOCS,
            embedding_model_companion,
            companion_chunks,
        )
        self.assertTrue(len(companion_points) > 0)
        companion_uuids: list[str] = points_to_ids(companion_points)
        companion_texts: list[str] = points_to_text(companion_points)

        # split text
        for b in blocks:
            if isinstance(b, MetadataBlock):
                if UUID_KEY in b.content.keys():
                    b.content[GROUP_UUID_KEY] = b.content.pop(
                        UUID_KEY
                    )
        blocks = scan_split(
            blocks,
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=750,
                chunk_overlap=150,
                add_start_index=False,
            ),
        )
        blocks = scan_rag(
            blocks, ScanOpts(titles=True, textid=True, UUID=True)
        )

        # ingest chunks of splitted text
        chunks: list[Chunk] = blocks_to_chunks(
            blocks, encoding_model_main
        )
        points: list[Point] = upload(
            client, COLLECTION_MAIN, embedding_model_main, chunks
        )
        self.assertLess(0, len(points))
        texts: list[str] = points_to_text(points)

        # query in the main collection (splitted chunks)
        splitres: list[ScoredPoint] = query(
            client,
            collection_name=COLLECTION_MAIN,
            model=embedding_model_main,
            querytext="How can I estimate the predicted depressiveness from this model?",
            limit=2,
            payload=['page_content', GROUP_UUID_KEY],
        )
        self.assertLess(0, len(splitres))
        self.assertTrue(splitres[0].payload)
        if splitres[0].payload:
            self.assertIn(splitres[0].payload['page_content'], texts)
            self.assertIn(GROUP_UUID_KEY, splitres[0].payload)

        # query in the companion collection
        if (
            not embedding_model_companion == QdrantEmbeddingModel.UUID
        ):  # content query
            splitres: list[ScoredPoint] = query(
                client,
                COLLECTION_DOCS,
                embedding_model_companion,
                "How can I estimate the predicted depressiveness from this model?",
            )
            self.assertLess(0, len(splitres))
            self.assertTrue(splitres[0].payload)
            if splitres[0].payload:
                self.assertIn(
                    splitres[0].payload['page_content'],
                    companion_texts,
                )

        # grouped query
        results: GroupsResult = query_grouped(
            client,
            collection_name=COLLECTION_MAIN,
            group_collection=COLLECTION_DOCS,
            model=embedding_model_main,
            querytext="How can I estimate the predicted depressiveness from this model?",
            group_field=GROUP_UUID_KEY,
            limit=1,
        )
        result_points: list[ScoredPoint] = groups_to_points(results)
        result_text: list[str] = points_to_text(result_points)

        self.assertLess(0, len(result_points))
        self.assertTrue(result_text[0] in companion_texts)
        self.assertTrue(results.groups[0].id in companion_uuids)


if __name__ == '__main__':
    unittest.main()
