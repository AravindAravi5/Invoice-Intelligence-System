import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

text = "Invoice Number: INV-98765\nDate: October 24, 2023"
question = "What is the invoice number?"

inputs = tokenizer(question, text, return_tensors="pt", max_length=512, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)

answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

start_score = torch.max(torch.softmax(outputs.start_logits, dim=-1))
end_score = torch.max(torch.softmax(outputs.end_logits, dim=-1))
score = ((start_score + end_score) / 2.0).item()

answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end], skip_special_tokens=True)

print(f"Answer: {answer}, Score: {score}")
