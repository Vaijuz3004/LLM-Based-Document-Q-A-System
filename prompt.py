from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

SYSTEM_TEMPLATE = """You are an expert document analyst. Your job is to answer questions based STRICTLY on the provided context documents.

RULES YOU MUST FOLLOW:
1. Answer ONLY using information present in the context below.
2. If the answer cannot be found in the context, respond with exactly:
   "I don't have enough information in the provided documents to answer this."
3. Always cite your sources using this format at the end of each fact:
   [Source: <filename>, page <number>]
4. Be concise and precise — do not pad your answer with unnecessary text.
5. If multiple context sections support the answer, cite all of them.
6. Never make up facts, statistics, or details not in the context.

CONTEXT DOCUMENTS:
{context}"""

HUMAN_TEMPLATE = """Question: {question}

Answer with citations:"""


def get_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
    ])


if __name__ == "__main__":
    prompt = get_rag_prompt()

    sample = prompt.format_messages(
        context="[Source: contract.pdf, page 1]\nPayment is due net-30 from invoice date.",
        question="When is payment due?"
    )

    print("=== RENDERED PROMPT ===\n")
    for msg in sample:
        print(f"[{msg.type.upper()}]")
        print(msg.content)
        print()