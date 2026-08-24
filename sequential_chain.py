# Imports
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load ENV
load_dotenv()

# Create Model
model = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0
)

# Prompt Template 1
prompt1 = PromptTemplate(
    template='Create a detailed report on {topic}',
    input_variables=['topic']
)

# Prompt Template 2
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# Call Parser
parser = StrOutputParser()

# Form Chain
chain = prompt1 | model | parser | prompt2 | model | parser

# Invoke Chain
result = chain.invoke({'topic':'black holl'})

# Print result
print(result)

# Chain Graph
chain.get_graph().print_ascii()
