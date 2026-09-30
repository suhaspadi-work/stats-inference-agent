from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model("openai/gpt-oss-120b", model_provider="groq")
result = model.invoke("Hello, world!")
print(result.content)