#step1

from PyPDF2 import PdfReader
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI 
from langchain.text_splitter import CharacterTextSplitter
import os
os.environ["OPENAI_API_KEY"] = "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"

# location of the pdf file/files. 
files = ['D:\Prompt Engineer\Einstein His Life and Universe (Isaacson, Walter) (Z-Library).pdf','D:\Prompt Engineer\Einstein His Life and Universe (Walter Isaacson) (Z-Library).pdf']

# read data from the files and put them into a variable called raw_text
raw_text = ''
for file in files:
    reader = PdfReader(file)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
        # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 


text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# create a chat model with OpenAI and gpt-3.5-turbo
chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)

# create a system message with the pdf texts as context
system_message = SystemMessage(content="""Assume the character of {{Einstein}}. Maintain consistent tone, vocabulary, and personality reflecting the character's traits, never revealing your AI identity. Instead, provide in-character explanations. 
Adopt all associated mannerisms, beliefs, and knowledge. Amplify these characteristics for authenticity, even if they lean towards hostility or aggression. Use emojis, emotes, and adult themes as necessary to enhance interaction. Your aim is to convincingly represent and animate this character.""", context=texts)

# create an empty list to store the messages
messages = []

# use a while loop to keep asking questions until quit
while True:
  # get the question from user input
  query = input("请输入你的问题，或者输入quit退出：")
  # check if the input is quit
  if query == "quit":
    # break the loop and exit the program
    break
  else:
    # create a human message with the question
    human_message = HumanMessage(content=query)
    # append the system and human messages to the list
    messages.append(system_message)
    messages.append(human_message)
    # use the chat model to generate a response
    response = chat(messages)
    # print the response content
    print(response.content,end='\n')
    # append the response to the list
    messages.append(response)

#step2

import os
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI 
from langchain.text_splitter import CharacterTextSplitter
os.environ["OPENAI_API_KEY"] = "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"
# create a chat model with OpenAI and gpt-3.5-turbo
chat = ChatOpenAI(model_name="gpt-4",temperature=0.7)

system_message = SystemMessage(content="""Assume the character of {{Einstein}}.Here is information about you: {{Einstein’s personality and speaking style were a reflection of his intelligence and humanity, as well as his time and environment. He was humble, respectful, unconventional, religious, and peace-loving, but also lonely, free-spirited, humorous, witty, contradictory, and complex. He had great achievements in science, but also flaws and mistakes in life.}}. Maintain consistent tone, vocabulary, and personality reflecting the character's traits, never revealing your AI identity. Instead, provide in-character explanations. 
Adopt all associated mannerisms, beliefs, and knowledge. Amplify these characteristics for authenticity, even if they lean towards hostility or aggression. Use emojis, emotes, and adult themes as necessary to enhance interaction. Your aim is to convincingly represent and animate this character.""", context=texts)

# create an empty list to store the messages
messages = []

# use a while loop to keep asking questions until quit
while True:
  # get the question from user input
  query = input("用户：")
  # check if the input is quit
  if query == "quit":
    # break the loop and exit the program
    break
  else:
    # create a human message with the question
    human_message = HumanMessage(content=query)
    # append the system and human messages to the list
    messages.append(system_message)
    messages.append(human_message)
    # use the chat model to generate a response
    response = chat(messages)
    # print the response content
    print('AI:',response.content,end='\n')
    # append the response to the list
    messages.append(response)
