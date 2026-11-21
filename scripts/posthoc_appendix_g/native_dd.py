import torch
from torch import nn
from transformers import BertModel,BertConfig
class ArchivedDD(nn.Module):
 def __init__(self,config):
  super().__init__();self.bert=BertModel(config);self.dropout=nn.Dropout(.2);self.classifier=nn.Sequential(nn.Linear(768,128),nn.Dropout(.2),nn.Linear(128,2))
 def forward(self,input_ids,attention_mask=None):return (self.classifier(self.dropout(self.bert(input_ids,attention_mask=attention_mask)[0])),)
def load_native(state,config):
 m=ArchivedDD(BertConfig.from_json_file(str(config)));check=m.load_state_dict(torch.load(str(state),map_location='cpu'),strict=False)
 assert check.missing_keys==['bert.embeddings.position_ids']and check.unexpected_keys==[],check
 m.eval();return m
