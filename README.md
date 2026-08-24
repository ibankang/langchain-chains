# LangChain Chains: A Practical Learning Repo

This repository is a hands-on collection of small LangChain programs that demonstrate how to compose language-model workflows with the **LangChain Expression Language (LCEL)**.

The examples are intentionally readable and progressively introduce core chain patterns:

- A simple prompt-to-model-to-parser pipeline
- Sequential chains, where one step feeds the next
- Parallel chains, where independent tasks run side by side before their results are merged
- Conditional chains, where structured model output determines the next branch

The goal is to show practical understanding of how LangChain runnables, prompts, models, parsers, and branching fit together.

## What This Repository Demonstrates

By working through these examples, you can practice:

- Building reusable prompts with `PromptTemplate`
- Connecting components with the LCEL pipe operator (`|`)
- Calling a chat model through `ChatGroq`
- Converting an `AIMessage` into plain text with `StrOutputParser`
- Passing outputs between multiple chain stages
- Running independent workflows with `RunnableParallel`
- Combining parallel outputs into a later prompt
- Defining validated structured output with Pydantic
- Routing execution with `RunnableBranch`
- Embedding normal Python behavior in a chain with `RunnableLambda`
- Inspecting workflows with `chain.get_graph().print_ascii()`
- Managing API credentials with environment variables and `python-dotenv`

## Repository Structure

| File | Main topic | What it shows |
| --- | --- | --- |
| [`simple_chain.py`](simple_chain.py) | Basic LCEL chain | Prompt a Groq chat model and parse its response as text. |
| [`sequential_chain.py`](sequential_chain.py) | Sequential composition | Generate a detailed report and pass it into a summary prompt. |
| [`parallel_chain.py`](parallel_chain.py) | Parallel execution | Generate notes and quiz questions independently, then merge both outputs. |
| [`conditional_chain.py`](conditional_chain.py) | Conditional routing | Classify feedback as positive or negative, then select the matching response chain. |
| [`requirements.txt`](requirements.txt) | Dependencies | Lists the Python packages used by the examples. |

## Concepts In Practice

### 1. Simple Chain

[`simple_chain.py`](simple_chain.py) creates the smallest complete LCEL workflow:

```text
input topic -> PromptTemplate -> ChatGroq -> StrOutputParser -> string output
```

The chain is invoked with a dictionary containing `topic`. The model generates five facts, and the string parser extracts usable text from the model response.

### 2. Sequential Chain

[`sequential_chain.py`](sequential_chain.py) connects two prompt/model/parser stages:

```text
 topic
   |
 report prompt -> model -> string parser
   |
 summary prompt -> model -> string parser
```

The first parser returns a string. LangChain then maps that value into the `{text}` variable required by the second prompt. This is useful for multi-step workflows such as research followed by summarization, extraction followed by transformation, or drafting followed by refinement.

### 3. Parallel Chain

[`parallel_chain.py`](parallel_chain.py) uses `RunnableParallel` to perform two independent tasks on the same source text:

```text
                 -> notes chain -\
source text -----                  -> merge prompt -> model -> final document
                 -> quiz chain  -/
```

The parallel runnable produces a dictionary with `notes` and `quiz`. That dictionary becomes the input to the merge prompt. This pattern is useful when tasks do not depend on one another and can be composed before a final synthesis step.

### 4. Conditional Chain

[`conditional_chain.py`](conditional_chain.py) demonstrates structured classification followed by conditional routing:

```text
feedback -> classifier prompt -> ChatGroq -> PydanticOutputParser
                                              |
                         positive or negative sentiment
                                              |
                                RunnableBranch
                                  /          \
                    positive response    negative response
```

The `Feedback` Pydantic model restricts the classifier output to the literals `positive` and `negative`. `PydanticOutputParser` validates the model response and returns a typed object. `RunnableBranch` reads `x.sentiment` and invokes the corresponding response chain.

## Prerequisites

- Python 3.10 or newer
- A Groq API key
- Internet access when running the model calls
- PowerShell, Command Prompt, or a Unix-like shell

The examples use these Groq model identifiers:

- `openai/gpt-oss-120b`
- `groq/compound-mini`

Model availability can change. If Groq no longer supports one of these identifiers for your account, update the `model` value in the relevant script.

## Setup

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit the real key. Keep `.env` listed in `.gitignore`, and use a secret manager or environment configuration when deploying an application.

## Run The Examples

With the virtual environment activated, run any script from the repository root:

```powershell
python .\simple_chain.py
python .\sequential_chain.py
python .\parallel_chain.py
python .\conditional_chain.py
```

Each script prints:

1. The generated result
2. An ASCII representation of its runnable graph

The model responses are generated dynamically, so the exact wording will vary even when `temperature=0` is used.

## Suggested Learning Order

1. Start with [`simple_chain.py`](simple_chain.py) and identify the input, prompt, model, parser, and final output.
2. Open [`sequential_chain.py`](sequential_chain.py) and trace how the first output becomes the second prompt's `text` input.
3. Study [`parallel_chain.py`](parallel_chain.py) to see why the intermediate result is a dictionary with named fields.
4. Finish with [`conditional_chain.py`](conditional_chain.py) and follow the transition from raw model output to a validated Pydantic object and then to a selected branch.
5. Modify the topics, feedback, prompts, and models, then observe how the graph and output change.

## How To Extend The Exercises

Possible next experiments include:

- Replace hard-coded inputs with command-line arguments.
- Add a `neutral` sentiment to the Pydantic schema and branch logic.
- Use `ChatPromptTemplate` for system and human messages.
- Add retries and error handling for API or parsing failures.
- Compare different models and temperatures.
- Make the sequential workflow accept the first result explicitly through `RunnablePassthrough` or a mapping.
- Add tracing and observability with LangSmith.
- Turn the examples into reusable functions or a small CLI application.
- Add tests using fake or mocked model responses so tests do not make live API calls.

## Important Notes

- These scripts execute their example chain when run; they are educational examples rather than packaged library modules.
- Running the scripts makes live requests to Groq and may consume API quota.
- Keep prompts and API credentials out of source control.
- The ASCII graph requires the `grandalf` dependency included in `requirements.txt`.
- Generated text should be reviewed before being treated as factual or customer-facing content.

## License

No license has been specified for this learning repository yet.
