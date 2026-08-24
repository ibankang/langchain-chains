from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 intersesting facts about {topic}',
    input_variables=['topic']
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii() # Print chain workflow graph