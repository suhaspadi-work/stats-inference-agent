from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def main():
    model = init_chat_model("openai/gpt-oss-120b", model_provider="groq")
    result = model.invoke("Hello, world!")
    print(result.content)


if __name__ == "__main__":
    main()
