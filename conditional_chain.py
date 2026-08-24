# ============================================================
# IMPORTS
# ============================================================

# Chat model integration for Groq
from langchain_groq import ChatGroq

# Converts AIMessage output into a normal Python string
from langchain_core.output_parsers import StrOutputParser

# Used to create prompt templates with variables such as {feedback}
from langchain_core.prompts import PromptTemplate

# RunnableBranch:
#   Used for conditional routing between different chains
#
# RunnableLambda:
#   Converts a normal Python function/lambda into a Runnable
#
# RunnableParallel:
#   Used for running multiple chains in parallel
#   NOTE: It is NOT used in this particular code.
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda
)

# Parses LLM output into a validated Pydantic object
from langchain_core.output_parsers import PydanticOutputParser

# Used to define the expected structured output
from pydantic import BaseModel, Field

# Literal restricts a value to specific allowed options
from typing import Literal

# Loads environment variables from the .env file
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

# Loads variables such as:
#
# GROQ_API_KEY=your_api_key
#
# from your .env file.
load_dotenv()


# ============================================================
# CREATE THE LLM
# ============================================================

model = ChatGroq(

    # The model that will perform the classification
    # and generate the final response
    model="openai/gpt-oss-120b",

    # temperature=0 makes the output more deterministic
    # and less random
    temperature=0
)


# ============================================================
# STRING OUTPUT PARSER
# ============================================================

# Chat models normally return an AIMessage object.
#
# Example:
# AIMessage(content="Thank you for your feedback!")
#
# StrOutputParser extracts only:
#
# "Thank you for your feedback!"
parser = StrOutputParser()


# ============================================================
# DEFINE THE STRUCTURED OUTPUT SCHEMA
# ============================================================

class Feedback(BaseModel):

    # The LLM is allowed to return ONLY one of these values:
    #
    # "positive"
    # "negative"
    #
    # Literal prevents values such as:
    #
    # "happy"
    # "neutral"
    # "good"
    sentiment: Literal["positive", "negative"] = Field(

        # This description is included in the output format
        # instructions given to the model
        description="Classify the sentiment of the feedback"
    )


# ============================================================
# CREATE THE PYDANTIC OUTPUT PARSER
# ============================================================

# This parser tells LangChain:
#
# "The model output should follow the Feedback schema"
#
# Expected result:
#
# Feedback(
#     sentiment="positive"
# )
#
# or:
#
# Feedback(
#     sentiment="negative"
# )
parser2 = PydanticOutputParser(
    pydantic_object=Feedback
)


# ============================================================
# CLASSIFICATION PROMPT
# ============================================================

prompt1 = PromptTemplate(

    # {feedback} will receive the user's feedback.
    #
    # {format_instruction} tells the model exactly how to
    # structure its response so that PydanticOutputParser
    # can parse it.
    template="""
Classify the sentiment of the following feedback text
as positive or negative.

Feedback:
{feedback}

{format_instruction}
""",

    # feedback must be provided when the chain is invoked
    input_variables=["feedback"],

    # This value is automatically inserted into the prompt.
    #
    # parser2.get_format_instructions() generates instructions
    # based on the Feedback Pydantic schema.
    partial_variables={
        "format_instruction": parser2.get_format_instructions()
    }
)


# ============================================================
# CREATE THE CLASSIFIER CHAIN
# ============================================================

# The | operator creates a RunnableSequence.
#
# Flow:
#
# Input
#   ↓
# prompt1
#   ↓
# model
#   ↓
# parser2
#
# The final result is NOT a string.
#
# It is a Pydantic object:
#
# Feedback(sentiment="positive")
#
# or:
#
# Feedback(sentiment="negative")
classifier_chain = (
    prompt1
    | model
    | parser2
)


# ============================================================
# POSITIVE RESPONSE PROMPT
# ============================================================

prompt2 = PromptTemplate(

    template="""
Write an appropriate response to this positive feedback:

{feedback}
""",

    input_variables=["feedback"]
)


# ============================================================
# NEGATIVE RESPONSE PROMPT
# ============================================================

prompt3 = PromptTemplate(

    template="""
Write an appropriate response to this negative feedback:

{feedback}
""",

    input_variables=["feedback"]
)


# ============================================================
# CREATE THE CONDITIONAL BRANCH
# ============================================================

branch_chain = RunnableBranch(

    # --------------------------------------------------------
    # CONDITION 1
    # --------------------------------------------------------
    #
    # x is the output of classifier_chain.
    #
    # Example:
    #
    # x = Feedback(sentiment="positive")
    #
    # If this condition is True:
    #
    #   x.sentiment == "positive"
    #
    # Then this chain is executed:
    #
    # prompt2
    #   ↓
    # model
    #   ↓
    # StrOutputParser
    (
        lambda x: x.sentiment == "positive",

        prompt2
        | model
        | parser
    ),


    # --------------------------------------------------------
    # CONDITION 2
    # --------------------------------------------------------
    #
    # If:
    #
    # x.sentiment == "negative"
    #
    # Then this chain is executed:
    #
    # prompt3
    #   ↓
    # model
    #   ↓
    # StrOutputParser
    (
        lambda x: x.sentiment == "negative",

        prompt3
        | model
        | parser
    ),


    # --------------------------------------------------------
    # DEFAULT BRANCH
    # --------------------------------------------------------
    #
    # If none of the previous conditions are True,
    # this RunnableLambda is executed.
    #
    # RunnableLambda allows us to use a normal
    # Python function inside an LCEL chain.
    RunnableLambda(
        lambda x: "Could not find sentiment"
    )
)


# ============================================================
# COMBINE BOTH CHAINS
# ============================================================

# The | operator connects the two chains.
#
# Overall flow:
#
# User Feedback
#      ↓
# classifier_chain
#      ↓
# Feedback(sentiment="positive" or "negative")
#      ↓
# branch_chain
#      ↓
# Positive or Negative response
#
chain = (
    classifier_chain
    | branch_chain
)


# ============================================================
# EXECUTE THE CHAIN
# ============================================================

result = chain.invoke({

    # Input enters classifier_chain
    "feedback": "This is a beautiful phone"
})


# Print the final generated response
print(result)


# ============================================================
# DISPLAY THE CHAIN GRAPH
# ============================================================

# get_graph() creates a graph representation of the complete chain.
#
# print_ascii() displays the chain structure in the terminal.
chain.get_graph().print_ascii()


# ============================================================
# DETAILED WORKING FLOW
# ============================================================

# 1. The user provides:
#
#    {
#        "feedback": "This is a beautiful phone"
#    }
#
#
# 2. The input goes to classifier_chain.
#
#
# 3. prompt1 creates a prompt similar to:
#
#    Classify the sentiment of the following feedback text
#    as positive or negative.
#
#    Feedback:
#    This is a beautiful phone
#
#    Return output in the required structured format.
#
#
# 4. The prompt is sent to the Groq model.
#
#
# 5. The model classifies the feedback as positive.
#
#    Example model output:
#
#    {
#        "sentiment": "positive"
#    }
#
#
# 6. PydanticOutputParser converts and validates that output.
#
#    The result becomes:
#
#    Feedback(
#        sentiment="positive"
#    )
#
#
# 7. This Pydantic object is passed to RunnableBranch.
#
#
# 8. RunnableBranch checks conditions from top to bottom.
#
#    First condition:
#
#    x.sentiment == "positive"
#
#    This is True.
#
#
# 9. RunnableBranch selects the positive branch:
#
#    prompt2 | model | parser
#
#
# 10. The original feedback is used to generate
#     a positive response.
#
#
# 11. The model returns an AIMessage.
#
#
# 12. StrOutputParser converts:
#
#     AIMessage(content="Thank you for your wonderful feedback!")
#
#     into:
#
#     "Thank you for your wonderful feedback!"
#
#
# 13. This final string is returned by:
#
#     print(chain.invoke(...))