# from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from Watermark_hidden_demo import HiddenMessageWatermarkLogitsProcessor, HiddenMessageWatermarkDetector
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)
import torch


model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map=f"cuda:1")

watermark_processor = HiddenMessageWatermarkLogitsProcessor(
    vocab=list(tokenizer.get_vocab().values()),
    delta=0.4,
    hidden_message=">e3zv",  # 
    bigram_table_path="wiki_bigram_table.pkl",
    entropy_skip_percentile=0.5,  # 30%
    message_length=5,
    probability_aware_greenlist=True,
)
# processor = RaptorWatermarkProcessor(
#     hidden_string="test",
#     bigram_table_path="wiki_bigram_table.pkl",
#     delta=20,  # 
#     # use_error_correction=True,
#     entropy_skip_percentile=0,
# )
# Note:
# You can turn off self-hashing by setting the seeding scheme to `minhash`.

original_text = "After the martyrdom of St. Boniface, Vergilius was made Bishop of Salzburg (766 or 767) and laboured successfully for the upbuilding of his diocese as well as for the spread of the Faith in neighbouring heathen countries, especially in Carinthia. He died at Salzburg, 27 November, 789."
tokenized_input = tokenizer(original_text, return_tensors='pt').to(model.device)
# note that if the model is on cuda, then the input is on cuda
# and thus the watermarking rng is cuda-based.
# This is a different generator than the cpu-based rng in pytorch!

output_tokens = model.generate(**tokenized_input,
                               logits_processor=LogitsProcessorList([watermark_processor]),min_new_tokens=400,max_new_tokens=500)

# if decoder only model, then we need to isolate the
# newly generated tokens as only those are watermarked, the input/prompt is not
output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]
# print(output_tokens)
output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

print(output_text)
detector = HiddenMessageWatermarkDetector(
    vocab=list(tokenizer.get_vocab().values()),
    device=model.device,
    tokenizer=tokenizer,
    bigram_table_path="wiki_bigram_table.pkl",
    entropy_skip_percentile=0.5,
    message_length=5,
    normalizers=[],
    probability_aware_greenlist=True,
)

# result = detector.detect_hidden_string(text=original_text)
# print(f"original_text:{result}")

result = detector.detect_hidden_message(text=output_text)
print(f"detect_hidden_text:{result['message']}")
