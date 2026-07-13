from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# Create a Model
gemini = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

deepseekv4 = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

# Prompt Template 1
prompt_1 = PromptTemplate(
    template = "Generate short and simple notes from the following text. \n {text}",
    input_variables= ['text']
)

# Prompt Template 2
prompt_2 = PromptTemplate(
    template = "Generate 5 short question answers from the following text. /n {text}",
    input_variables= ['text']
)

# Prompt Template 3
prompt_3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document. Notes -> {notes} and Quiz -> {quiz}",
    input_variables= ['notes', 'quiz']
)

parallel_chain = RunnableParallel({
    'notes': prompt_1 | deepseekv4 | parser,
    'quiz' : prompt_2 | gemini | parser
})

merging_chain = prompt_3 | gemini | parser

main_chain = parallel_chain | merging_chain

result = main_chain.invoke({
    'text' : '''Hi everyone! I’ve been diving deep into LangChain for the past two months, and now that I’ve explored a lot of concepts, I felt it was the right time to start sharing my learnings here. I’m revisiting everything with documenting it properly through this blog series. I hope you enjoy the content and find it useful!

Before we talk about Runnables, it’s important to understand what a Chain is, because Runnables build on top of the same idea but in a much more flexible and modern way.

What is a Chain in LangChain?
A Chain is a core component in LangChain that allows engineers to build pipelines connecting multiple tasks together.

Press enter or click to view image in full size

Let’s look at a very simple example where we take the user’s input, pass it to an LLM, and then clean up the LLM’s output before returning it.

1. User enters a prompt
It all begins when the user types something like:

“Explain vector databases in one sentence.”

This becomes the raw input to the chain.

2. The LLM thinks and responds
The prompt is passed to the LLM, which generates a natural-language answer.
For example:

“A vector database is designed to store and search high-dimensional embeddings efficiently.”

Straightforward and simple.

3. We clean the output using a Structured Output Parser
Instead of returning the raw text as-is, we pass the LLM’s output through a StrOutputParser.

This helps in:

removing unnecessary whitespace,
ensuring consistent formatting,
returning a clean and predictable string.
A basic chain is essentially this:

input → LLM → parser → final output

It keeps the process organized and makes the pipeline easier to manage.

LangChain supports three major types of Chains (which I’ll cover later in the Runnable Primitive section):

Sequential Chains
Parallel Chains
Branch Chains
Now that we have a clear understanding of Chains, let’s move to Runnables.

What is a Runnable in LangChain?
Runnables are one of the most important concepts introduced in newer versions of LangChain.
You can think of them as a unified abstraction that lets you connect all kinds of components LLMs, prompts, retrievers, tools, parsers, functions into a single pipeline.

And the best part?
You don’t need to remember separate methods for each component.

Why were Runnables introduced?
Earlier, every component in LangChain had its own style:

Prompts had format()
LLMs had predict() or generate()
Chains had run() or invoke()
Parsers had parse()
This made things confusing and forced engineers to remember multiple method names.'''
})

print(result)

