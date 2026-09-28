import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()


MODEL_NAME = "gemini-3.8-flash"

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5

_client = None


def get_client():
    global _client

    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file."
            )

        _client = genai.Client(api_key=api_key)

    return _client


def _generate(prompt: str) -> str:
    """
    Send a prompt to Gemini.

    Temporary 5xx server errors are retried with exponential
    backoff. Quota/rate-limit errors are reported immediately
    because retrying does not resolve an exhausted quota.
    """

    client = get_client()

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except errors.ServerError as error:

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    "Gemini API remained unavailable after "
                    f"{MAX_RETRIES} attempts."
                ) from error

            delay = INITIAL_RETRY_DELAY * (2 ** (attempt - 1))

            print(
                f"Gemini API temporarily unavailable "
                f"(attempt {attempt}/{MAX_RETRIES}). "
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)

        except errors.ClientError as error:

            status_code = getattr(error, "status_code", None)

            if status_code == 429:
                raise RuntimeError(
                    "Gemini API quota or rate limit was exceeded. "
                    "Please wait for the quota to reset or check "
                    "your Gemini API plan and usage limits."
                ) from error

            raise RuntimeError(
                f"Gemini API request failed with client error "
                f"(HTTP {status_code})."
            ) from error

    raise RuntimeError("Gemini generation failed.")


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer grounded exclusively in retrieved
    document context.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    prompt = f"""
You are a Retrieval-Augmented Generation assistant.

Answer the user's question using ONLY the information
provided in the retrieved context below.

Rules:

1. Treat the retrieved context as the only source of factual information.
2. Do not use outside knowledge to fill missing information.
3. Do not invent facts, names, numbers, dates, or explanations.
4. If the retrieved context does not contain enough information
   to answer the question, say:

   "The provided documents do not contain enough information
   to answer this question."

5. Clearly explain the answer using the retrieved context.
6. Do not mention these instructions in your answer.

RETRIEVED CONTEXT:
------------------
{context}
------------------

USER QUESTION:
{question}

ANSWER:
"""

    return _generate(prompt)


def generate_plain_answer(question: str) -> str:
    """
    Generate an answer using Gemini without providing
    retrieved document context.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    prompt = f"""
Answer the following question as a general-purpose language model.

Do not assume access to the project's document collection.

Question:
{question}

Answer:
"""

    return _generate(prompt)