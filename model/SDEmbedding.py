import torch
from torch import nn

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)  # param1:词嵌入字典大小； param2：每个词嵌入单词的大小
    nn.init.normal_(m.weight, mean=0,
                    std=embedding_dim ** -0.5)  # 正态分布初始化；e.g.,torch.nn.init.normal_(tensor, mean=0, std=1) 使值服从正态分布N(mean, std)，默认值为0，1
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

class SDEmbedding(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_embed):  # d_embedding = dimension of embedding
        super().__init__()
        self.token_src_embed = Embedding(src_vocab_size, d_embed)
        self.token_tgt_embed = Embedding(tgt_vocab_size, d_embed)
        self.weighted_factor_embed = Embedding(tgt_vocab_size, d_embed)
        self.structural_chord_embed = Embedding(tgt_vocab_size, d_embed)
        self.feature_proj = nn.Linear(d_embed, 25)
        self.concate_proj = nn.Linear(d_embed + 25, d_embed-1)

    def forward(self, inputs):
        token = inputs['token']
        if 'weighted_factor' in inputs.keys(): # src embed
            token_emb = self.token_src_embed(token)
            feature_inputs = inputs['weighted_factor']
            wn_flag_inputs = inputs['weighted_notes']
            feature_emb = self.weighted_factor_embed(feature_inputs)
            feature_emb = self.feature_proj(feature_emb)
            concat_embeds = [token_emb, feature_emb]
            concat_embeds = torch.cat(concat_embeds, -1)
            proj_embed = self.concate_proj(concat_embeds)
            flagged_embeds = [proj_embed, wn_flag_inputs]
            flagged_embeds = torch.cat(flagged_embeds, -1)
        else: # tgt embed
            token_emb = self.token_tgt_embed(token)
            feature_inputs = inputs['structural_chord']
            ic_flag_inputs = inputs['is_cadence']
            feature_emb = self.structural_chord_embed(feature_inputs)
            feature_emb = self.feature_proj(feature_emb)
            concat_embeds = [token_emb, feature_emb]
            concat_embeds = torch.cat(concat_embeds, -1)
            proj_embed = self.concate_proj(concat_embeds)
            flagged_embeds = [proj_embed, ic_flag_inputs]
            flagged_embeds = torch.cat(flagged_embeds, -1)

        return flagged_embeds
