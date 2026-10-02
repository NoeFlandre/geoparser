# Concepts

This page explains the ideas that the library uses. It also explains the terms that the documentation uses. It has no code. The [glossary](glossary.md) gives short definitions.

## The Problem

Text has many places. Examples are a news archive, a corpus of interviews, a collection of travel diaries, or a set of historical records. They all refer to where events occurred. But this geography is in prose. You cannot map it, count it, or join it to other data. For a computer, "Basel" is only five letters.

**Geoparsing** is the process that changes these mentions into geographic data. You give it a text. It gives you a list of the places that the text mentions. It ties each place to a specific point or area on the Earth.

Geoparsing has two problems. They are different problems.

## Recognition: which words are places?

The first task is to find the place names. Read this sentence:

> *The delegation flew from Basel to Santiago in March.*

A person sees immediately that "Basel" and "Santiago" are places. "March" is not a place. All three words start with a capital letter. To do this task automatically is a known problem in natural language processing. It is called **named entity recognition**. A *toponym* is a place name. It is one type of named entity.

This task is more difficult than a match against a list of known place names. There are two reasons. First, many place names are also ordinary words or other types of name. Reading is a town and a verb. Jordan is a country and a surname. Turkey is a country and a bird. Second, a list is never complete. For this reason, recognition uses **context**. The context is the grammar and the vocabulary around a word. It does not use a lookup. The approaches range from rules written by hand to statistical models and neural models. Today, a trained language model usually does the task. For this reason, recognition can fail on text that is not like the training text of the model.

In this library, a **recognizer** does recognition. Its output is a set of character positions. These are spans of the text that appear to be place names. At this stage, nothing is geographic yet.

## Resolution: which place is it?

The second task is to decide which place each name refers to.

GeoNames has 122 places with the name Paris. 45 of them are in the United States. It has 291 places with the name Springfield. The text "Santiago" can mean the capital of Chile. It can mean one of several cities in Cuba or Spain. It can also mean a person. The correct choice is called **toponym resolution**. Other names for it are toponym disambiguation and geocoding.

Again, the answer is in the context. Read these sentences:

> *The delegation flew from Basel to Santiago in March.*
>
> *Pilgrims have walked to Santiago for a thousand years.*

The name is the same, but the two cities are different. You know which city is which from the words around the name. You do not know it from the name. In the second sentence, "pilgrims" and "thousand years" point to Santiago de Compostela in Spain. Nothing in the first sentence points to it. The capital of Chile is the more prominent city.

A **resolver** does this task. For each recognized name, it looks up the candidates. It compares them with the context of the mention. It selects one candidate. If no candidate is convincing, it selects none. This is important. The default resolver of the library leaves a toponym unresolved when no candidate passes its similarity threshold. Thus unresolved toponyms are a normal part of the output.

## Gazetteers: the list of candidates

To choose between candidate places, you need a list of the places that exist. This list is a **gazetteer**. A gazetteer is a database of places. Each entry has these items:

- a stable **identifier**,
- one or more **names**, which include historical and foreign-language forms,
- usually a **location**, which is a point or an area,
- a set of **attributes**. These are the type of the place, the places that it contains or that contain it, the number of people who live there, and so on.

The attributes make resolution possible. GeoNames describes one Santiago as the capital of Chile with 4.8 million inhabitants. It describes another Santiago as the seat of a Spanish region in Galicia with about 100,000 inhabitants. A resolver compares the words around a mention with descriptions like these. A gazetteer records plain factual data of this type. It does not tell you that one of these cities is a pilgrimage destination. If you can identify a mention only with knowledge that the gazetteer does not have, the case is difficult.

For this reason, you must install a gazetteer to set up the library. This is a required step. Recognition works on text only. But to resolve a name to a place in this library, the resolver selects an entry from a gazetteer. A gazetteer must exist. Other methods exist that predict coordinates directly from the words. They do not use a list of candidate places. This library selects from a gazetteer. Thus you get an identifier and attributes. You do not get only a pair of coordinates.

The gazetteer that you choose also has a larger effect on your results than any other decision. A gazetteer with only cities cannot resolve a river. A gazetteer for the modern world only cannot resolve a Roman province. This is true for all models. The page [querying gazetteers](guides/gazetteers.md) describes what the pre-configured gazetteers contain. If neither gazetteer describes the world that you study, build a gazetteer from your own data. Read the [custom gazetteers guide](guides/custom-gazetteers.md).

## Putting It Together

A pipeline in this library has two stages in sequence. The text goes to the recognizer. The recognizer returns the spans of the place names that it found. The spans and the text go to the resolver. The resolver returns the place that each name refers to. The gazetteer is not a third stage. It is the resource that the resolver uses. It belongs to the resolver. When you construct the resolver, you tell it which gazetteer to use.

When you build a `Geoparser`, you state both modules explicitly. There are no defaults for them.

This is deliberate. The two stages are independent. To separate them is the central design decision of the library. You can change how names are found and not change how they are resolved. You can compare two recognizers with the same resolver and see which finds more. You can give a resolver a different gazetteer. You can replace a stage with your own implementation. The page [configuring modules](guides/modules.md) describes the existing modules and how to write a module.

The two stages can also fail independently. Remember this when a result looks wrong. If nothing comes back, first check if the recognizer found anything. A resolver cannot resolve a name that a recognizer did not find.

## What Comes Out

The results have the same shape as the pipeline that made them.

A **document** is one text that you give to the library. Each document has **toponyms**. These are the place names that the library found in it. Each toponym has a `text`. It also has the position in the document (`start` and `end` character offsets). You use the position to get back to the sentence around the name. Each toponym also has a `location`. This is the gazetteer entry that the resolver selected. If the resolver did not resolve the toponym, the `location` is `None`.

A location is a **feature**. A feature is one entry in a gazetteer. It has the `data` of the gazetteer for that place as a dictionary. It has its `geometry` as a shape. You can map, measure, or export the shape. Gazetteers are different, so the available attributes are different. GeoNames calls the name of a place `name`. SwissNames3D calls it `NAME`. Do not assume the attribute names. Read them defensively.

## Two Ways to Work

The library has two entry points for the same pipeline. The only difference is if the library keeps the results.

`Geoparser` is the direct entry point. Text goes in and documents come out. The library stores nothing. Use it for an analysis that you do in one session, in a script or a notebook. The results go directly to your next step. The [quickstart](quickstart.md) uses it. Start with this entry point.

`Project` is a persistent workspace. It stores the documents and the results in a database under a name. You can process a corpus once and come back next week. The results are still there. You can also run several different pipelines on the same corpus. You keep all their outputs side by side under different **tags**. This makes a systematic comparison possible. The texts are the same. The configurations are different. You can count the results against each other. Use it for corpus research. Use it to build and evaluate annotated data. Use it for results that you do not want to compute again. Read [managing projects](guides/projects.md).

Start with `Geoparser`. Change to `Project` when you want to keep results.

> [!NOTE]
> We expect the `Geoparser` interface to stay stable. The project layer, which uses a database, will probably become one option of several. It will no longer be the foundation of everything.

## Words You Will See

| Term            | Meaning                                                                                                                      |
|-----------------|------------------------------------------------------------------------------------------------------------------------------|
| **Toponym**     | A place name in a text. "Basel" in a sentence is a toponym. The city is not a toponym.                                        |
| **Reference**   | The name of this library for a recognized toponym. It is a span of text that the recognizer identified as a place name. `document.toponyms` returns these. |
| **Referent**    | The place that a reference points to. The library records it as a gazetteer name and an identifier.                          |
| **Feature**     | One entry in a gazetteer. It is a place with its names, attributes, and geometry. `toponym.location` gives it to you.        |
| **Recognition** | The task to find place names in text. Stage one.                                                                             |
| **Resolution**  | The task to decide which place each name refers to. Stage two. Other names are disambiguation and geocoding.                  |
| **Gazetteer**   | A database of places. The resolver uses it as the set of candidates.                                                         |
| **Module**      | A recognizer or a resolver. The parts of a pipeline that you can replace.                                                    |
| **Tag**         | A label for the results of one pipeline in a project. Several sets of results can exist for the same documents.             |
| **Artifact**    | The single self-contained file that an installed gazetteer is.                                                               |
