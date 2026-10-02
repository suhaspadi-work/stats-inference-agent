import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# A system prompt roughly the size/shape we'd expect for the stats agent:
# tool descriptions + instructions, not a trivial one-liner
system_prompt = """
You are a statistical inference agent. You have access to tools for:
- profiling datasets (row/column counts, types, missingness, duplicates, cardinality)
- wrangling operations (cast, rename, filter, dedupe, join, aggregate, derive)
- hypothesis testing (two-group tests, multi-group tests, assumption checks)
- regression (linear and logistic, with diagnostics)

You must never compute a statistic yourself. Always call the appropriate tool.
You must classify each request by question type and data situation, produce a
structured analysis plan, and freeze that plan before executing any test.
You must not use causal language ("causes", "drives", "leads to") unless the
plan's causal_status explicitly allows it — default to "associated with".
""".strip()

response = client.chat.completions.with_raw_response.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Say 'ok' and nothing else."},
    ],
)

parsed = response.parse()
print("Response:", parsed.choices[0].message.content)
print()
print("=== Rate limit headers ===")
for key in [
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
]:
    print(f"{key}: {response.headers.get(key)}")
print("\n=== Burst test: 5 calls in a row ===")
for i in range(5):
    r = client.chat.completions.with_raw_response.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Reply with just the number {i}."},
        ],
    )
    remaining = r.headers.get("x-ratelimit-remaining-tokens")
    reset = r.headers.get("x-ratelimit-reset-tokens")
    print(f"call {i}: remaining_tokens={remaining}, reset_in={reset}")