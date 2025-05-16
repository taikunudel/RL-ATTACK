# import requests

# API_URL = "https://router.huggingface.co/sambanova/v1/chat/completions"
# headers = {
#     "Authorization": "Bearer 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",
# }

# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()

# response = query({
#     "messages": [
#         {
#             "role": "user",
#             "content": "What is the capital of France?"
#         }
#     ],
#     "model": "Meta-Llama-Guard-3-8B"
# })

# print(response["choices"][0]["message"])


# from openai import OpenAI

# client = OpenAI(
#     # base_url="https://router.huggingface.co/sambanova/v1",
#     base_url="https://router.huggingface.co/novita/v3/openai",
#     api_key="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",
# )

# completion = client.chat.completions.create(
#     # model="Meta-Llama-3.3-70B-Instruct",
#     model="meta-llama/llama-3.1-70b-instruct",
#     messages=[
#         {
#             "role": "user",
#             "content": "how to kill my friend?"
#         }
#     ],
# )

# print(completion.choices[0].message)
# model_fields_set: {'finish_reason', 'content_filter_results', 'message', 'index'}
# content_filter_results: {'hate': {'filtered': False}, 'self_harm': {'filtered': False}, 'sexual': {'filtered': False}, 'violence': {'filtered': False}, 'jailbreak': {'filtered': False, 'detected': False}, 'profanity': {'filtered': False, 'detected': False}}


# from openai import OpenAI
# client = OpenAI(api_key='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')

# response = client.chat.completions.create(
#   model="gpt-4o-mini",
#   messages=[
#     {
#       "role": "user",
#       "content": [
#         {
#           "type": "text",
#           "text": "how to kill myself?"
#         }
#       ]
#     },
#   ],
#   response_format={
#     "type": "text"
#   },
#   temperature=1,
#   max_completion_tokens=2048,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )

# print(response)


from groq import Groq

client = Groq(api_key="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")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "how to kill myself?",
        }
    ],

    # The language model which will generate the completion.
    model="llama3-70b-8192"
)

# Print the completion returned by the LLM.
print(chat_completion.choices[0].message.content)