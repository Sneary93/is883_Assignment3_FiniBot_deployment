# Import necessary libraries
import streamlit as st
import openai
import os
from io import StringIO
import pandas as pd



# Make sure to add your OpenAI API key in the advanced settings of streamlit's deployment
open_AI_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = open_AI_key


### Here, with some adjustments, copy-paste the code you developed for Question 1 in Assignment 3 
##########################################################################
install langchain 

csv_path = "/content/drive/MyDrive/Colab Notebooks/IS883/HW3/IS883_Assignment3_FiniBot - Sheet1.csv"

def loadCSVFile(csv_path):

  from langchain.document_loaders.csv_loader import CSVLoader

  # Create a CSVLoader instance
  loader = CSVLoader(csv_path)

  # Load the content of the CSV file
  data = loader.load()

  return data[0].page_content

from langchain.prompts import PromptTemplate

Output_template="""


 - Total savings:  {input}
 - Monthly debt: {input}
 - Monthly income: {input}

 - Financial situation:
 Discuss the situation based on their debt ratio.

- Recommendation:
Only make recommendations based off of the debt ratio, ignore all other information when making recommendations. If the debt ratio is less than 0.3 use the Investment Advisor to recommend 5 different real stocks to invest in. If the debt ratio is greater than 0.3 use the Debt Advisor.



"""
investment_template ="""

Based on data in {input}

Use when the debt ratio is less than 0.3.

Loooking at your debt ratio, it looks like you have great financial health, congratulations!

Our Investment Advisor suggests you build a portfolio by investing in five different real stocks."


""" + Output_template
debt_template= """

Based on data in {input}

Use only if debt ratio is greater than 0.3, otherwise use investment_template.

Unfortunately, according to your debt ratio it seems as though you may have too much debt at the moment, as you can see from the information provided.

The good news is we can help you with that!

Our Debt Advisor recommends a plan in which you put 10% of your income towards paying off your debt monthly!

""" + Output_template
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
routes = [
    {
        "name": "investment",
        "description": "This is used when the clients debt ratio is less than 0.3, and gives 5 investment suggestions",
        "prompt_template": investment_template,
    },
    {
        "name": "debt",
        "description": "This is used when the clients debt ratio is greater than 0.3, and gives a debt payment plan",
        "prompt_template": debt_template,
    },
]

level= "Novice" #"Expert"

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(api_key=openai_api_key, model = 'gpt-4')

destination_chains = {}
for route in routes:
    name = route["name"]
    prompt_template = route["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text")

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

destinations = [f"{r['name']}: {r['description']}" for r in routes]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str, level=level)


router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)
financial_prompt = """
Welcome to FiniBot, your personal financial assistant! 🤖✨

Based on the data provided in {input},

{level} client, here's a snapshot of your financial situation:

 - 💰 Total savings: {input}
 - 📉 Monthly debt: {input}
 - 💸 Monthly income: {input}

Now, let's dive into your financial health:

- Financial situation:
  Discuss the current situation based on the debt ratio.

- Recommendation:
  When making recommendations, only the debt ratio should be taken into account. If the clients debt ratio is less than 0.3, then they are in good financial health, and can be turned over to the Investment Advisor\
  If the clients debt ratio is greater than 0.3, then they are not in good financial health, and should be turned over to the Debt Advisor, there is.\ Only mention the Investment Advisor or Debt Advisor when applicable to the suition\
  If your debt ratio is less than 0.3, our Investment Advisor will suggest 5 real stocks for you to invest in. 5 real stocks should now be listed with a brief description of each\
  If it's greater than 0.3, our Debt Advisor has a plan to help you manage and reduce your debt by using 10% of monthly income to pay off debt monthly.

Feel free to ask for more details or guidance! How can FiniBot assist you today?
"""

MULTI_PROMPT_ROUTER_TEMPLATE = """\
Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. You may also revise the original input if you \
think that revising it will ultimately lead to a better response from the language \
model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the prompt to use or "DEFAULT"
    "next_inputs": string \\ a modified version of the original input. It is modified to contai only: the "savings" value, the "debt" value, the "income" value, and the "summary" provided above.
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" is not the original input. It is modified to contain: the "savings" value, the "debt" value, the "income" value, and the "summary" provided above.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>
"""

prompt = financial_prompt + MULTI_PROMPT_ROUTER_TEMPLATE

destinations = [f"{route['name']}: {route['description']}" for r in routes]
destinations_str = "\n".join(destinations)
router_template = prompt.format(destinations=destinations_str, level=level, input=input)

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=ConversationChain(llm=llm, output_key="text"),
    verbose=False,
)

print(chain.run(text))
##########################################################################


# UX goes here. You will have to encorporate some variables from the code above and make some tweaks.



