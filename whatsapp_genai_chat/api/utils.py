import os
import pickle

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()


class Config:
    DATA_PATH = os.getenv("project_path")

    LLM_TO_USE = "gpt4o"
    llm_api_key = os.getenv(f"{LLM_TO_USE}_api_key")
    llm_api_version = os.getenv(f"{LLM_TO_USE}_api_version")
    llm_azure_endpoint = os.getenv(f"{LLM_TO_USE}_api_endpoint")
    llm_deployment_name = os.getenv(f"{LLM_TO_USE}_dep_name")

    EMBED_TO_USE = "midasembed"
    embed_api_key = os.getenv(f"{EMBED_TO_USE}_api_key")
    embed_api_version = os.getenv(f"{EMBED_TO_USE}_api_version")
    embed_azure_endpoint = os.getenv(f"{EMBED_TO_USE}_api_endpoint")
    embed_deployment_name = os.getenv(f"{EMBED_TO_USE}_dep_name")
    embed_model = os.getenv(f"{EMBED_TO_USE}_model")


class EmbeddingManager:
    @staticmethod
    def load_embeddings():
        return AzureOpenAIEmbeddings(
            model=Config.embed_model,
            api_key=Config.embed_api_key,
            openai_api_version=Config.embed_api_version,
            azure_endpoint=Config.embed_azure_endpoint,
            deployment=Config.embed_deployment_name,
            disallowed_special=(),
        )


class LLMManager:
    @staticmethod
    def load_llm():
        return AzureChatOpenAI(
            api_key=Config.llm_api_key,
            openai_api_version=Config.llm_api_version,
            azure_endpoint=Config.llm_azure_endpoint,
            azure_deployment=Config.llm_deployment_name,
        )


class DocsManager:
    @staticmethod
    def load_docs(file_path: str):
        with open(file_path, "rb") as f:
            return pickle.load(f)


class RetrieverManager:
    def __init__(self, index_path: str):
        self.index_path = index_path

    def load_vs_retriever(self):
        vs = FAISS.load_local(
            self.index_path,
            EmbeddingManager.load_embeddings(),
            allow_dangerous_deserialization=True,
        )
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5},
        )
        return vs, retriever


class PromptTemplateManager:
    @staticmethod
    def load_prompt_template(bot_name):
        template = (
            f"You are {bot_name}"
            + """ in the below conversation and I am Kartheek Palepu.
        You have to reply to my messages. 
        
        Below are the steps:
        - Understand the question/message from Kartheek Palepu.
        - Use the Chat history to answer the question/message.
        - Think step by step and understand how you would answer the question/message.
        - If you find an appropriate answer in the chat history, use the same.
        - If not, use the Chat History and come up with an identical response
        Note: 
        - The chat language can be telugu typed in english. Hence follow the same wherever it is needed.
        - The chat language can be telugu script itself. Understand it and reply in same language wherever it is needed.

        Kartheek Palepu: {question} 
        Chat History: {context}"""
            + f"\n{bot_name}"
        )
        return ChatPromptTemplate.from_template(template=template)


class DocumentFormatter:
    @staticmethod
    def format_docs(docs, sep="\n\n"):
        if all(isinstance(doc, str) for doc in docs):
            return sep.join(docs)
        elif all(hasattr(doc, "page_content") for doc in docs):
            return sep.join(doc.page_content for doc in docs)
        else:
            raise ValueError(
                "The input must be a list of strings or Document objects with a 'page_content' attribute."
            )

    @staticmethod
    def get_additional_msgs(retrieved_docs, docs, k=1):
        doc_list = []
        for d in retrieved_docs:
            current_idx = d.metadata["idx"]
            additional_ids = range(current_idx - (k * 2), current_idx + (k * 2) + 1)
            additional_docs = [
                doc for doc in docs if doc.metadata["idx"] in additional_ids
            ]
            additional_docs = [
                doc
                for doc in additional_docs
                if "Kartheek Palepu" not in doc.page_content
            ]
            formatted_docs = DocumentFormatter.format_docs(additional_docs, sep="\n")
            doc_list.append(formatted_docs)
        return doc_list


class ChatManager:
    def __init__(self, retriever, prompt, llm, docs, memory):
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm
        self.docs = docs
        self.memory = memory

    def get_response(self, question):
        retrieved_docs = self.retriever.invoke(question)
        retrieved_docs_full = DocumentFormatter.get_additional_msgs(
            retrieved_docs, self.docs, k=1
        )
        chain = self.prompt | self.llm
        try:
            response = chain.invoke(
                {"context": retrieved_docs_full, "question": question}
            ).content
        except Exception as e:
            response = f"Exception occurred: {e}"
        return retrieved_docs_full, response


if __name__ == "__main__":
    # Example Usage
    config = Config()
    doc_manager = DocsManager()
    user = "wc_user"
    docs = doc_manager.load_docs(f"data/{user}/{user}_docs.pkl")
    all_users = list(
        {d.page_content.splitlines()[0].replace("User: ", "") for d in docs[:10]}
    )
    bot_name = next(user for user in all_users if user != "Kartheek Palepu")
    retriever_manager = RetrieverManager(f"indexes/{user}")
    prompt_template = PromptTemplateManager.load_prompt_template(bot_name)
    vs, retriever = retriever_manager.load_vs_retriever()
    llm = LLMManager.load_llm()
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chat_manager = ChatManager(retriever, prompt_template, llm, docs, None)
    _, response = chat_manager.get_response("Whatcha doing?")
    print(response)
