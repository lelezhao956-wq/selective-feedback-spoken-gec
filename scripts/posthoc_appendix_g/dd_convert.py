"""Run in original sandi-dd env. Converts checkpoint to tensor state; never fits or modifies it."""
from pathlib import Path
import torch,json
from transformers import BertTokenizerFast
P=Path(__file__).resolve().parents[4]/'work/v6';P.mkdir(exist_ok=True,parents=True)
m=torch.load('~/Desktop/sandi-corpus-2025-2/sandi-models/dd/dd_model',map_location='cpu');assert not m.training
torch.save(m.state_dict(),P/'dd_state.pt');(P/'dd_config.json').write_text(m.config.to_json_string())
t=BertTokenizerFast.from_pretrained('bert-base-uncased',is_pretokenized=True,local_files_only=True);t.save_pretrained(str(P/'dd_tokenizer'))
print('CHECKPOINT_CONVERTED_WITHOUT_FITTING',P,'classifier',str(m.classifier))
