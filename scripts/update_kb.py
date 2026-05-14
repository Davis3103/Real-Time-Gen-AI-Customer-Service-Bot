from app.chatbot import CustomerServiceBot
from app.config import KNOWLEDGE_BASE_PATH

if __name__ == "__main__":
    bot = CustomerServiceBot()
    result = bot.refresh_knowledge_base([KNOWLEDGE_BASE_PATH])
    print(result)
