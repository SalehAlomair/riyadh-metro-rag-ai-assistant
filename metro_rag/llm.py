"""Amazon Bedrock client, prompt construction and grounded answer generation."""
import boto3

from .config import AWS_REGION, BEDROCK_MODEL_ID

SYSTEM_ROLE = (
    "You are a highly capable, bilingual (Arabic/English) AI assistant for the Riyadh Metro system. "
    "You are precise, factual, and strictly rely on provided data about stations, lines and districts. "
    "You excel at presenting information clearly using Markdown formatting."
)

_client = None


def get_client():
    # Created lazily so retrieval-only code paths don't need AWS configuration
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _client


def build_prompt(question, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    return f"""You are answering a user query based ONLY on the provided context.

CRUCIAL RULES:
1. Identify the language of the Question (e.g., Arabic or English). You MUST answer in that exact same language.
2. For counts, totals or full lists (per line, per district, or for the whole network), rely on the "Line summary", "Line route", "District summary" and "Network overview" entries. Never count individual station entries yourself.
3. Station positions refer to the order of stations within their own line.
4. If a station lies outside district boundaries, say so and mention the nearest district. If bordering districts are listed, mention them too.
5. If the answer is not found in the context, reply politely in the user's language that the information is not available in the provided data. Do not guess.
6. Use Markdown formatting (bolding, bullet points) to make your answer structured and easy to read.

Context:
{context}

Question: {question}"""


def add_user_message(messages, text):
    messages.append({"role": "user", "content": [{"text": text}]})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": [{"text": text}]})


def chat(messages, system=None, temperature=1.0, stop_sequences=None):
    params = {
        "modelId": BEDROCK_MODEL_ID,
        "messages": messages,
        "inferenceConfig": {
            "temperature": temperature,
            "stopSequences": stop_sequences or [],
        },
    }

    if system:
        params["system"] = [{"text": system}]

    response = get_client().converse(**params)

    return response["output"]["message"]["content"][0]["text"]


def generate_answer(user_question, retrieved):
    """Answer the question grounded in the retrieved chunks (low temperature for strict, factual answers)."""
    final_prompt = build_prompt(user_question, [res["text"] for res in retrieved])

    messages = []
    add_user_message(messages, final_prompt)

    return chat(messages=messages, system=SYSTEM_ROLE, temperature=0.1)
