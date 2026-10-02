# Configuring Modules

Modules are the parts of a pipeline that you can replace. This guide describes the two built-in kinds of module. It also describes the parameters that matter when your results are poor, and how to write your own module.

## What a Module Is

There are two kinds of module. A **recognizer** finds place names in text. It takes texts and returns character positions. A **resolver** links those names to places. It takes texts and positions and returns gazetteer entries. The two kinds are independent of each other. A pipeline has one recognizer and one resolver.

The interface is small on purpose. A module receives text and returns predictions. Other parts of the library do the rest: they store results, they prevent duplicate work, and they keep different runs apart. Thus a module is easy to write. Implement one method, and the module works everywhere that the built-in modules work. The [modules API](../api/modules.md) reference gives the full signatures of all modules.

The library identifies a module by its class **and its configuration**. `SpacyRecognizer()` and `SpacyRecognizer(model_name="en_core_web_trf")` are two different modules. They have separate results. Because of this, you can run a pipeline again without repeated work, and you can compare configurations. But if you change a parameter, the library does not update your old results. It makes new results and keeps the old results.

## Built-in Recognizers

### GLiNER2Recognizer

The `GLiNER2Recognizer` uses [GLiNER2.5](https://huggingface.co/fastino/gliner2.5-multi-v1). This is a zero-shot extractor. You give the entity types as ordinary words when you call it. The model does not contain a fixed set of types. Therefore the recognizer works with many languages immediately. You can also tune the label list for your corpus.

Use it with the default settings:

``` python
from geoparser.modules import GLiNER2Recognizer

recognizer = GLiNER2Recognizer()
```

The default configuration uses the `fastino/gliner2.5-multi-v1` checkpoint. It looks for `city`, `country`, and `location`. You can change both settings:

``` python
from geoparser.modules import GLiNER2Recognizer

recognizer = GLiNER2Recognizer(
    entity_types=["city", "country", "river", "mountain", "national park"],
)
```

The recognizer matches the labels zero-shot. Therefore it is usually better to name the types that you need than to use a bigger model. For example, assume your corpus is about hiking routes. If you ask for `mountain` and `trail`, you find things that a fixed GPE/LOC/FAC schema does not show. But each extra label adds inference time. The recognizer also removes duplicates from overlapping labels. If a span matches `city` and `country`, it becomes one reference and not two.

The recognizer merges the spans from all labels into one list. The list is in the order of the positions in the text. Thus the resolver gets the references in the order in which they occur.

For texts that are longer than 10,000 characters, the recognizer uses the long-document mode of GLiNER2. This mode scans chunks of a fixed number of words. It keeps memory use low for long documents and for noisy OCR text, where each character can make several tokens. The recognizer processes shorter texts as one piece. The [`GLiNER2Recognizer` API page](../api/modules.md) describes the same behavior.

### SpacyRecognizer

The `SpacyRecognizer` uses the named entity recognition of spaCy to find possible place names in text. By default, it accepts entities with the labels geopolitical entity (GPE), location (LOC), and facility (FAC) as toponyms. You can change this.

Use the SpacyRecognizer with the default settings:

``` python
from geoparser.modules import SpacyRecognizer

recognizer = SpacyRecognizer()
```

The default configuration uses the `en_core_web_sm` model and the entity types FAC, GPE, and LOC. You can change both parameters:

``` python
from geoparser.modules import SpacyRecognizer

# Use a more accurate transformer-based model
recognizer = SpacyRecognizer(
    model_name="en_core_web_trf",
    entity_types=["GPE", "LOC"],  # Only geopolitical entities and locations
)
```

The `model_name` parameter accepts any spaCy model that has a named entity recognizer. Larger models, such as `en_core_web_trf`, are more accurate. But they need more memory and more processing time. For texts that are not in English, specify a spaCy model for that language.

The `en_core_web_trf` pipeline needs the `spacy-curated-transformers` plugin. Install the compatible pinned line with `pip install "spacy-curated-transformers>=0.3.1,<1"`. The plugin has no Python 3.14 release yet. On that interpreter, use a non-transformer model for the requested language, for example `en_core_web_lg` for English.

The `entity_types` parameter sets which entity types the recognizer accepts as toponyms. By default, it includes FAC (facilities, such as buildings and landmarks), GPE (geopolitical entities, such as countries and cities), and LOC (natural locations and regions). If your application needs only country names and city names, restrict the types to GPE.

## Built-in Resolvers

### PriorResolver

`PriorResolver` uses the same MiniLM encoder, the same gazetteer search, and the same context windows as `SentenceTransformerResolver`. It adds a population score when it ranks close candidates. The score of a candidate is its context similarity plus `population_weight * log10(1 + population) / 10`. A candidate without a usable population gets no bonus. The default weight is `0.3`. The `min_similarity` threshold checks the raw context similarity before the ranking. Thus the population cannot lift a weak match over the threshold.

The inflection fallback is off by default. Set `inflection_fallback=True` to try again after an exact miss. The resolver then removes one to three trailing characters from a single-word name. This can help with declined or inflected place names. It does not change phrase searches, partial searches, or fuzzy searches.

``` python
from geoparser.modules import PriorResolver

resolver = PriorResolver(
    population_weight=0.3,
    inflection_fallback=False,
)
```

Refer to the [benchmark guide](benchmark.md) for the comparison results and the ablation results. Refer to the [resolver API](../api/modules.md#resolvers) for all constructor arguments.

### JinaResolver

The `JinaResolver` extends the `SentenceTransformerResolver` (see below). It changes how the resolver selects a candidate. It keeps the same tiered gazetteer search and the same context windows. It is different in two ways.

First, it makes embeddings with [jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small). This model has **separate prompts for the two sides of a retrieval pair**. The resolver embeds the context of a reference as a query. It embeds the description of a candidate as a document. This is how the developers trained the model. If you embed both sides with one prompt, you lose most of the benefit.

Second, it adds a **reranking stage**. The embedding comparison is fast, but it looks at each side alone. Therefore the resolver uses it only to make a shortlist. Then [jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) selects the winner from the shortlist. This model is a cross encoder. It reads the context and the candidate description together.

``` python
from geoparser.modules import JinaResolver

resolver = JinaResolver()
```

The defaults are the two checkpoints above, a shortlist of 20 candidates, the `geonames` gazetteer, a minimum similarity of 0.6, and a maximum of 3 search tiers. These parameters are specific to this resolver:

``` python
from geoparser.modules import JinaResolver

resolver = JinaResolver(
    rerank_top_k=50,  # rerank more candidates, more slowly
    gazetteer_name="swissnames3d",
)
```

`rerank_top_k` is the parameter that you must understand. The cross encoder is much slower for each candidate than the embedding comparison. The shortlist keeps the resolution time acceptable. A wider shortlist helps when the embeddings rank the correct candidate outside the top 20. But it increases the time in proportion. `min_similarity` still applies to the embedding stage. If the best candidate of a reference does not reach it, the resolver leaves the reference unresolved. It does not use the reranker.

### SentenceTransformerResolver

The `SentenceTransformerResolver` uses transformer language models to disambiguate place names. It compares contextual embeddings. It takes the context around each place name and gets the candidate locations from the gazetteer. It makes a text description of each candidate. Then it selects the candidate whose description is closest to the context by embedding cosine similarity.

Use the SentenceTransformerResolver with the default settings:

``` python
from geoparser.modules import SentenceTransformerResolver

resolver = SentenceTransformerResolver()
```

The default configuration uses the `dguzh/geo-all-MiniLM-L6-v2` model, the `geonames` gazetteer, and a minimum similarity threshold of 0.6. It expands through a maximum of 3 tiers of search methods that become wider each time. You can change all of these parameters:

``` python
from geoparser.modules import SentenceTransformerResolver

# Use a more accurate model with Swiss gazetteer
resolver = SentenceTransformerResolver(
    model_name="dguzh/geo-all-distilroberta-v1",
    gazetteer_name="swissnames3d",
    min_similarity=0.5,  # Accept more candidates
    max_tiers=2,  # Search through fewer tiers for faster processing
)
```

The `model_name` parameter sets the SentenceTransformer model that makes the embeddings. The library has two pre-trained models, fine-tuned for toponym disambiguation. `dguzh/geo-all-MiniLM-L6-v2` is fast and has good accuracy. `dguzh/geo-all-distilroberta-v1` is more accurate but uses more time and memory. The developers trained these models on English news articles. They work best with English text and the GeoNames gazetteer. For other languages or domains, train a custom model. Refer to the [training](training.md) guide.

The `gazetteer_name` parameter sets the geographic database to search. The gazetteer must be installed on your system. Each gazetteer has different coverage and different attribute schemas. Make sure that the gazetteer satisfies the requirements of your application.

The `min_similarity` threshold sets how confident the resolver must be before it accepts a match. A higher threshold gives fewer false positives, but more toponyms stay unresolved. A lower threshold resolves more toponyms, but it can add incorrect matches. If no candidate reaches the threshold, the toponym stays unresolved. Thus the resolver prefers precision to recall.

The `max_tiers` parameter sets how widely the resolver searches for candidates. The resolver uses an iterative strategy. It starts with exact string matching. Then it uses phrase matching, partial matching, and fuzzy matching. For each search method, it ranks the results by relevance and puts them into tiers. `max_tiers` sets how many of these tiers the resolver includes. A higher value gives more possible candidates. This can help to resolve difficult toponyms, but it increases the processing time.

The `attribute_map` parameter tells the resolver how to read the attributes of the gazetteer. Before the resolver compares a candidate with the text, it describes the candidate in words, for example "Paris (city) in Île-de-France, France". Each gazetteer uses different names for its attributes. Thus the resolver must know which attributes to use for the sentence. The resolver already knows the mapping for GeoNames and SwissNames3D. Give the parameter only for a gazetteer that you made yourself:

``` python
from geoparser.modules import SentenceTransformerResolver

resolver = SentenceTransformerResolver(
    gazetteer_name="my_gazetteer",
    attribute_map={
        "name": "place_name",
        "type": "category",
        "level1": "country",
        "level2": "region",
        "level3": "district",
    },
)
```

The values are keys of the `data` dictionary of your gazetteer. The configuration of the gazetteer sets which keys exist (refer to [custom gazetteers](custom-gazetteers.md)). `name` and `type` are required. The administrative levels are optional. `level1` is the outermost enclosing place and `level3` is the innermost. Give only as many levels as your data has. For example, a gazetteer of ancient places can have nothing above the Roman province of a place:

``` python
resolver = SentenceTransformerResolver(
    gazetteer_name="pleiades",
    attribute_map={
        "name": "title",
        "type": "place_types",
        "level1": "province",
    },
)
```

This map describes a candidate as "Pompeii (settlement, urban area) in Italia". If a key is not in the `data` of a feature, the resolver omits it from the description and does not fail. Thus a mapping can name an attribute that only some of your features have.

Note that the developers fine-tuned the pre-trained models on GeoNames-style descriptions. If the vocabulary of your gazetteer is very different, lower `min_similarity`. For the best results, fine-tune your own resolver on data that is annotated with the features of that gazetteer. Refer to [training](training.md).

The SentenceTransformerResolver works best when place names have distinctive contexts. For example, "I visited the Eiffel Tower in Paris" gives strong clues. Short texts with little context are more difficult. Lists of place names without surrounding text are also more difficult. The resolver can fail in these cases.

## Creating Custom Recognizers

To make a custom recognizer, implement the `Recognizer` interface. A recognizer is a class that inherits from `Recognizer`. It implements a `predict()` method. The method takes a list of texts and returns the predictions for each text.

This is the basic structure of a custom recognizer:

``` python
import typing as t
from geoparser.modules.recognizers import Recognizer


class MyCustomRecognizer(Recognizer):
    NAME = "MyCustomRecognizer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your recognizer here
        # Store any configuration parameters

    def predict(
        self, texts: t.List[str]
    ) -> t.List[t.Union[t.List[t.Tuple[int, int]], None]]:
        # Implement your recognition logic here
        # Return list of reference positions for each text
        pass
```

The `NAME` class attribute gives a name that a person can read for your recognizer. The `__init__` method must call the parent initializer. Give the configuration parameters as keyword arguments. The library stores these parameters in the configuration of the module and uses them to make its unique ID.

The `predict()` method receives a list of document texts. It must return a list of the same length. For each document, return one of two values. Return a list of `(start, end)` tuples with the character positions of the place names that the recognizer found. Or return `None` if the recognizer cannot process that document, for example if the document is in an unsupported language.

This is a complete example of a simple recognizer that uses regular expressions:

``` python
import typing as t
import re
from geoparser.modules.recognizers import Recognizer


class RegexRecognizer(Recognizer):
    """Recognizer that identifies place names using regular expressions."""

    NAME = "RegexRecognizer"

    def __init__(self, patterns: t.List[str]):
        """
        Initialize with a list of regex patterns.

        Args:
            patterns: Regular expressions that match place names
        """
        super().__init__(patterns=patterns)
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def predict(
        self, texts: t.List[str]
    ) -> t.List[t.Union[t.List[t.Tuple[int, int]], None]]:
        """Find all matches of the patterns in each text."""
        results = []

        for text in texts:
            references = []

            # Find all matches for each pattern
            for pattern in self.patterns:
                for match in pattern.finditer(text):
                    references.append((match.start(), match.end()))

            # Sort by start position and remove overlaps
            references.sort()
            results.append(references)

        return results
```

You can use this custom recognizer in the same way as the built-in recognizers:

``` python
from geoparser import Project

# Create recognizer that looks for country names
recognizer = RegexRecognizer(
    patterns=[
        r"\b(France|Germany|Italy|Spain|Switzerland)\b",
        r"\b(United States|United Kingdom|New Zealand)\b",
    ]
)

project = Project("regex_test")
project.create_documents(["I traveled from France to Germany."])
project.run_recognizer(recognizer)
```

When you implement a custom recognizer, make sure that the `(start, end)` positions are real character offsets in the text. When possible, align them with token boundaries or entity boundaries. Overlapping references can cause problems in later processing. Remove or merge them in your implementation.

## Creating Custom Resolvers

A custom resolver follows the same pattern, but it implements the `Resolver` interface. A resolver takes texts and reference positions as input. It returns the resolved referents (pairs of gazetteer name and identifier) for each reference.

This is the basic structure of a custom resolver:

``` python
import typing as t
from geoparser.modules.resolvers import Resolver


class MyCustomResolver(Resolver):
    NAME = "MyCustomResolver"

    def __init__(self, gazetteer_name: str, **kwargs):
        super().__init__(gazetteer_name=gazetteer_name, **kwargs)
        self.gazetteer_name = gazetteer_name
        # Initialize your resolver here

    def predict(
        self, texts: t.List[str], references: t.List[t.List[t.Tuple[int, int]]]
    ) -> t.List[t.List[t.Union[t.Tuple[str, str], None]]]:
        # Implement your resolution logic here
        # Return list of (gazetteer_name, identifier) tuples
        pass
```

The `predict()` method receives two lists. `texts` has the document texts. `references` has the reference positions for each document. The method must return a nested list with the same structure as `references`. Each element is a `(gazetteer_name, identifier)` tuple that points to a feature in the gazetteer. If the resolver cannot resolve the reference, the element is `None`.

Resolvers usually use gazetteers to find candidate locations. The library has the `Gazetteer` class for this task:

``` python
import typing as t
from geoparser.modules.resolvers import Resolver
from geoparser import Gazetteer


class PopulationResolver(Resolver):
    """Resolver that selects the most populous candidate."""

    NAME = "PopulationResolver"

    def __init__(self, gazetteer_name: str = "geonames"):
        super().__init__(gazetteer_name=gazetteer_name)
        self.gazetteer_name = gazetteer_name
        self.gazetteer = Gazetteer(gazetteer_name)

    def predict(
        self, texts: t.List[str], references: t.List[t.List[t.Tuple[int, int]]]
    ) -> t.List[t.List[t.Union[t.Tuple[str, str], None]]]:
        """Resolve each reference to the most populous candidate."""
        results = []

        for text, doc_refs in zip(texts, references):
            doc_results = []

            for start, end in doc_refs:
                reference_text = text[start:end]

                # Search for candidates
                candidates = self.gazetteer.search(
                    reference_text, method="partial", limit=100
                )

                if candidates:
                    # Select candidate with highest population
                    best = max(
                        candidates, key=lambda c: c.data.get("population", 0) or 0
                    )
                    doc_results.append((self.gazetteer_name, best.identifier))
                else:
                    doc_results.append(None)

            results.append(doc_results)

        return results
```

The `Gazetteer` class has two main methods to get candidates. The `search()` method takes a place name string. It returns the matching features for the search method that you specify (`"exact"`, `"phrase"`, `"partial"`, or `"fuzzy"`). The `find()` method looks up a feature by its identifier. Refer to the `gazetteers` guide for more information.

When you implement a custom resolver, always handle the case with no candidates. Return `None` for that reference. Make sure that the returned structure is exactly the same as the `references` structure. Each document must have the same number of results as references, in the same order.

## Making Modules Trainable

To make a custom module trainable, implement a `fit()` method with the correct interface. For recognizers, the `fit()` method must accept texts and reference positions:

``` python
def fit(
    self, texts: t.List[str], references: t.List[t.List[t.Tuple[int, int]]], **kwargs
) -> None:
    """Train the recognizer on annotated data."""
    # Implement your training logic here
    pass
```

For resolvers, the `fit()` method must also accept referents:

``` python
def fit(
    self,
    texts: t.List[str],
    references: t.List[t.List[t.Tuple[int, int]]],
    referents: t.List[t.List[t.Optional[t.Tuple[str, str]]]],
    **kwargs,
) -> None:
    """Train the resolver on annotated data."""
    # Implement your training logic here
    pass
```

The `fit()` method can accept more keyword arguments for training parameters, such as the learning rate, the batch size, or the number of epochs. After you implement the method, you can train your custom modules with the project-level training methods. Refer to the [training](training.md) guide.
