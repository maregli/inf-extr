from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src import paths

# Download model
checkpoint = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model)

# Save model
model.save_pretrained(paths.MODEL_PATH/'llama2-chat')

# Save tokenizer
tokenizer.save_pretrained(paths.MODEL_PATH/'llama2-chat')