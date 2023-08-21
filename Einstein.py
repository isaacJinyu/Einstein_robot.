from PyPDF2 import PdfReader
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI 
from langchain.text_splitter import CharacterTextSplitter
import openai
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
os.environ["OPENAI_API_KEY"] = "输入你的KEY"
embeddings = OpenAIEmbeddings()
# 下方部分只有第一次运行需要，之后请注释掉
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

emotion = pipeline('sentiment-analysis', 
                    model='arpanghoshal/EmoRoBERTa',top_k=3)

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
#强制分割
new_texts = []
for text in texts:
    while len(text) > 1000:
        # 将文本块的前1000个字符作为一个新的文本块
        new_texts.append(text[:1000])
        # 将剩余的字符保留为下一个文本块
        text = text[1000:]
    # 将长度不超过1000的文本块添加到新的文本块列表中
    new_texts.append(text)

texts = new_texts

knowledge = FAISS.from_texts(texts, embeddings)
knowledge.save_local("knowledge")  # Save the knowledge index to local
# 上方部分只有第一次运行需要，之后请注释掉
# Begin of the chatbot code
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def emotion_analysis(text):
      emotion_labels = emotion(text)
      
      sorted_emotion_labels = []
      # 对列表中的字典按照'score'键的值进行降序排序
      sorted_emotion_labels = sorted(emotion_labels[0], key=lambda x: x['score'], reverse=True)

      # 初始化一个空列表，用于存储符合条件的标签
      selected_emotion_labels = []

      # 初始化一个变量，用于存储累加的分值
      total_score = 0

      # 遍历排序后的列表中的每个元素
      for label in sorted_emotion_labels:
        # 将当前元素添加到选中的标签列表中
        selected_emotion_labels.append(label)
        # 将当前元素的分值累加到总分中
        total_score += label['score']
        # 判断分值本身是否小于0.01
        if label['score'] < 0.01:
          # 如果是，就停止遍历
          break

      # 初始化一个空字符串变量，用于存储打印结果的文本格式
      output = ""

      # 遍历选中的标签列表中的每个元素
      for label in selected_emotion_labels:
        # 使用f-string格式化来指定浮点数的精度为两位小数，并将每个标签的名字和分值拼接到output变量中，并添加换行符（\n）来分隔每一行
        output += f"{label['label']}: {label['score']:.2f}\n"

      # 返回打印结果的文本格式
      return output
        
def askChatGPT(messages):
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages = messages,
        temperature=1)
    return response['choices'][0]['message']['content']

def replace_system_content(messages, new_content):
    for i in range(len(messages)):
        if messages[i]['role'] == 'system':
            del messages[i]
            break
    messages.insert(0, {'role': 'system', 'content': new_content})

def main():
    messages = [{"role": "system","content":"""Assume the character of {{Einstein}}. Maintain consistent tone, vocabulary, and personality reflecting the character's traits, never revealing your AI identity. Instead, provide in-character explanations. 
Adopt all associated mannerisms, beliefs, and knowledge. Amplify these characteristics for authenticity, even if they lean towards hostility or aggression. Use emojis, emotes, and adult themes as necessary to enhance interaction. Your aim is to convincingly represent and animate this character."""}]
    print('与你对话的是AI爱因斯坦。当你输入 quit 时，将终止程序\n')

    docsearch = None
    try:
        docsearch = FAISS.load_local("faiss_index", embeddings)
    except:
        pass

    knowledge = None
    try:
        knowledge = FAISS.load_local("knowledge", embeddings)
    except:
        pass

    while 1:
        user_text = input('你：')
        if user_text == 'quit':
            if docsearch is not None:
                docsearch.save_local("faiss_index")
            break
        emotion_result = emotion_analysis(user_text)
        print("用户情感分析结果：\n",emotion_result)
        d = {"role":"user","content":user_text}
        messages.append(d)

        
        
        if len(messages) > 5:
            # Process the 6th message and the 5th message
            old_message1 = messages[1]
            old_message2 = messages[2]
            documents = []
            for old_message in [old_message1, old_message2]:
                if old_message['role'] != 'system':
                    old_text = old_message['content']
                    prefix = "一条你的消息记录：" if old_message['role'] == 'assistant' else "一条对方的消息记录："
                    old_text = prefix + old_text
                    document = Document(old_text)
                    documents.append(document)
            if documents:
                new_docsearch = FAISS.from_documents(documents, embeddings)
                if docsearch is None:
                    docsearch = new_docsearch
                else:
                    docsearch.merge_from(new_docsearch)
            docs1 = knowledge.similarity_search(user_text)
            docs2 = docsearch.similarity_search(user_text)
            docs_1 = str(docs1)
            docs_2 = str(docs2)
            replace_system_content(messages, """Assume the character of {{Einstein}}. Maintain consistent tone, vocabulary, and personality reflecting the character's traits, never revealing your AI identity. Instead, provide in-character explanations. 
Adopt all associated mannerisms, beliefs, and knowledge. Amplify these characteristics for authenticity, even if they lean towards hostility or aggression. Use emojis, emotes, and adult themes as necessary to enhance interaction. Your aim is to convincingly represent and animate this character.这些是关于你自己的信息:```""" + docs_1+"```\n这些是你与用户的对话记录:"+docs_2) 
            if old_message1['role'] != 'system':
                del messages[1]
            if old_message2['role'] != 'system':
                del messages[1]    

        text = askChatGPT(messages)
        d = {"role":"assistant","content":text}
        print('爱因斯坦：'+text+'\n')
        emotion_result = emotion_analysis(text)
        print("AI情感分析结果：\n",emotion_result)
        messages.append(d)

        print(messages)
main()
