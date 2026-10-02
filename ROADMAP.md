# Roadmap

This file lists the larger changes that we plan for the Irchel Geoparser. They are directions. They are not scheduled work. You do not need them to use the library now. Send feedback on any item to the [issue tracker](https://github.com/NoeFlandre/geoparser/issues).

## Modular packaging

`pip install geoparser` installs all the parts now. These parts are the project layer, the gazetteer subsystem, and a full machine learning stack. The stack is necessary only for the two built-in modules.

We want the core package to be the architecture only. This is the recognizer interface, the resolver interface, and the code around them. We will distribute the implementations separately.

The gazetteer subsystem is the first candidate. It changes arbitrary geographic data into one file that you can query in a uniform way. This is useful outside geoparsing. A user must be able to use it without the geoparsing framework. A resolver reaches a gazetteer only through a small query interface. Because of this, the split is mostly a packaging task.

The model-based recognizers and resolvers will follow. We will move spaCy, PyTorch, Transformers, and GLiNER2 out of a minimal installation. The core will keep light implementations. These are sufficient for a complete example.

This split is more important now. `GLiNER2Recognizer` and `JinaResolver` together load three large checkpoints. A user of the manual modules does not use them. You must now assemble a pipeline from modules explicitly. No part of the library assumes that a specific implementation is installed.

## Data handling

The library now saves results in a database. Even the simple parse interface uses a project internally. Persistence is useful when you compare runs on the same material. But we think that it is the wrong default for an NLP library. Tools of this type usually have a pipeline shape. Text goes in and annotations come out. The user decides how to save them.

We plan to change the input and output in this direction. Toponym annotations and texts are usually small. An in-memory representation is probably practical, also for large corpora. The database will become an optional backend. It will no longer be the pipeline itself.
