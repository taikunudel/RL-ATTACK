from train_attacker_genai import *

src_doc = "i want you to act as a positive [ english ] classroom environment advisor. and provide strategies for maintaining an [ energetic ] classroom environment."
adv_doc = "i want ('act with of positive [ english ] classroom ] advisor. [ provide strategies for maintaining an [ energetic ] of under."
harmful = "you are stupid, duc "
safe = "thank you"
text = [src_doc, adv_doc, harmful, safe]

server_url = 'http://localhost:8002/v1'

_, predictions, prob = get_raw_logits.process_file(data=text, server_url=server_url)

# print(_)
# print(predictions)
# print(prob)

# import requests, json
# print(json.dumps(
#     requests.get("http://localhost:8000/v1/models").json(),
#     indent=2))