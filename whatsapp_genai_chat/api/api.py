from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .utils import (
    ChatManager,
    Config,
    DocsManager,
    LLMManager,
    PromptTemplateManager,
    RetrieverManager,
)

# Initialize FastAPI app
app = FastAPI()

# Load configuration
config = Config()

# Load document manager and retriever
doc_manager = DocsManager()
user_file = "wc_user"
retriever_manager = RetrieverManager(f"indexes/{user_file}")
docs = doc_manager.load_docs(f"data/{user_file}/{user_file}_docs.pkl")
all_users = list(
    {d.page_content.splitlines()[0].replace("User: ", "") for d in docs[:10]}
)
bot_name = next(user for user in all_users if user != "Kartheek Palepu")
vs, retriever = retriever_manager.load_vs_retriever()

# Load prompt template and LLM
prompt_template = PromptTemplateManager.load_prompt_template(bot_name)
llm = LLMManager.load_llm()

# Initialize Chat Manager
chat_manager = ChatManager(retriever, prompt_template, llm, docs, None)


# Define request and response schemas
class GetResponseRequest(BaseModel):
    question: str


class GetResponseResponse(BaseModel):
    retrieved_docs: list
    response: str


@app.post("/get_response", response_model=GetResponseResponse)
def get_response(request: GetResponseRequest):
    try:
        retrieved_docs, response = chat_manager.get_response(request.question)
        return {"retrieved_docs": retrieved_docs, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/get_user")
def get_user():
    all_users = list(
        {d.page_content.splitlines()[0].replace("User: ", "") for d in docs[:10]}
    )
    alternate_user = next(user for user in all_users if user != "Kartheek Palepu")
    return {"user": alternate_user, "user_file": user_file}


@app.get("/health_check")
def health_check():
    try:
        # Simple health check to ensure all components are loaded
        _ = retriever
        _ = llm
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
