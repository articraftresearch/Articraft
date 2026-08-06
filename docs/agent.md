# Agent design

mini-articraft has a small generation system. This is its core loop:

```text
prompt -> model -> workspace -> record
```

The model changes one Python workspace. The workspace compiles it in a separate process. The agent adds each model response and tool result to the conversation record.

## Run sequence

The command line interface and the `mini_articraft.generate` functions create a model adapter and
a local workspace. They give both items to `Agent`. This separation keeps the loop independent
of one model or compiler.

The agent does these steps for each run:

1. It creates a run directory and a starter `main.py` file.
2. It loads the system prompt, the SDK quickstart, and the user request.
3. It asks the model for text and tool calls.
4. It runs the tools and gives the results to the model.
5. It repeats the loop until the current workspace compiles.
6. It waits for a visible final response from the model.
7. It saves the final status and the USDZ path.

The loop has a turn limit. It also stops after three empty model responses. A run fails if the
model stops before a current compile. A run also fails if the compile does not make a USDZ file.

## Prompts and tools

The system prompt defines the authoring contract and the quality checks. It tells the model to
use correct dimensions and model the main motion. It tells the model to connect each part and
prevent unwanted overlap.

The SDK quickstart is a separate user message. Thus, the model starts with the current public
API.

The model has these local tools:

- `read` reads workspace files and the SDK reference.
- `view_image` opens workspace images and SDK reference figures.
- `edit` and `write` change workspace files.
- `exec_command` and `write_stdin` run short local inspections.
- `compile` builds, checks, and exports the current object.

Independent read operations can run at the same time. File changes, shell commands, and compile
operations run in sequence. Tool paths stay in the run workspace. Read operations can also use
the packaged SDK documents.

## Compile contract

The compile worker runs in a separate Python process. This process isolates failures in generated
code. It also gives each normal compile a clean Python interpreter.

The workspace entry point must define these items:

```python
def build_object_model() -> ArticulatedObject: ...

object_model = build_object_model()

def run_tests() -> TestReport: ...
```

The worker loads `main.py`. It runs the authored tests and the compiler tests. It then exports the
USDZ file. The worker returns a short set of compile signals to the model. The command line
interface and the run files contain the full result.

The agent saves a digest after each successful compile. A workspace change makes that compile
old. Thus, the agent cannot publish an old USDZ file after a new edit fails.

## Run files

A run has this structure:

```text
runs/<run-id>/
  conversation.jsonl
  record.json
  workspace/
    main.py
    docs/sdk -> packaged SDK docs
  result/
```

`conversation.jsonl` contains each model message and tool result in sequence. `record.json`
contains the status, compile attempts, result path, cost, and token use. The `result` directory
contains the numbered export files.

## Main parts of the code

- [`api.py`](../src/mini_articraft/api.py) contains the public synchronous and asynchronous
  generation functions and the provider routing they share with the command line interface.
- [`agent/`](../src/mini_articraft/agent) contains the turn loop and tools.
- [`agent/provider/`](../src/mini_articraft/agent/provider) contains the model adapters.
- [`agent/workspace/`](../src/mini_articraft/agent/workspace) creates runs and compiles them.
- [`compiler/`](../src/mini_articraft/compiler) is the compile itself, its result, and the signals
  the agent reads. It runs with or without an agent.
- [`sdk/`](../src/mini_articraft/sdk) contains object authoring, tests, and export code.
- [`agent/record.py`](../src/mini_articraft/agent/record.py) saves the run record and conversation
  log.
- [`prompts/`](../src/mini_articraft/prompts) contains the model instructions.

The `Model` and `Workspace` protocols are the two main extension points. A model adapter answers
queries and closes its resources. A workspace creates a run and compiles it -- both halves vary
together, because whatever machine holds the run directory is the machine that compiles it.

## Test the loop

The test harness can replace paid model calls with scripted responses. It can keep a compile worker
open for fast tests. It can also replay a saved model conversation. These options test the full
loop. The production environment still uses a new process for each compile.

Read the [test environment guide](../tests/README.md) for more information.
