# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers

"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, Tensor
from model.position_encoding import build_position_encoding
from model.resnet import build_backbone, MyResnet
from torch.nn.parameter import Parameter
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding


class MyTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = MyResnet(args)
        pthfile = args.pretrained_dir
        state_dict = torch.load(pthfile)
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self.backbone.load_state_dict(state_dict, False)

        self.transformer = build_transformer(args)
        self.class_embed = nn.Linear(args.hidden_dim, args.class_number)
        # self.position_embedding = build_position_encoding(args.d_model, args.src_len, args.pos_emb_type)
        self.position_embedding = BartLearnedPositionalEmbedding(args.max_position_embedding, args.d_model)
        self.Embedding = nn.Embedding(args.class_number, args.d_model)
        self.d_model = args.d_model
        self.src_len = args.src_len
        self.hidden_dim = args.hidden_dim
        # self.start = Parameter(torch.rand(self.hidden_dim), requires_grad=False)

    def pack(self, embedding, intensity_embedding, target, target_embedding):
        bs, spb, hd = embedding.size()
        start = intensity_embedding[:, -1, :].unsqueeze(1)  # bs x 1 x d_model
        # start = self.start.repeat(bs, 1, 1).to(target.device)  # bs x 1 x d_model
        target_key_embedding = torch.cat((start, target_embedding), dim=1)  # bs x src_len x d_model
        # pos_embed = self.position_embedding(embedding)
        # target_pos_embed = self.position_embedding(target_key_embedding)
        pos_embed = self.position_embedding(embedding.size()[:2]).unsqueeze(0).repeat(bs, 1, 1)
        target_pos_embed = self.position_embedding(target_key_embedding.size()[:2]).unsqueeze(0).repeat(bs, 1, 1)
        return embedding, intensity_embedding, target_embedding, target_key_embedding, pos_embed, target_pos_embed

    def permute(self, embedding, intensity_embedding, target_embedding, target_key_embedding, pos_embed,
                target_pos_embed):
        embedding = embedding.permute(1, 0, 2)
        intensity_embedding = intensity_embedding.permute(1, 0, 2)
        target_embedding = target_embedding.permute(1, 0, 2)
        target_key_embedding = target_key_embedding.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        target_pos_embed = target_pos_embed.permute(1, 0, 2)
        return embedding, intensity_embedding, target_embedding, target_key_embedding, pos_embed, target_pos_embed

    def forward(self, x: Tensor, intensity, target, test_mode_target=None):
        """
        :param x: bs x src_len x C x H x W
        :param intensity: bs x src_len
        :param target: bs x src_len
        :param test_mode_target: batch_size x src_len
        :return:
        """
        embedding = self.backbone(x)  # bs x src_len x d_model
        bs, src_len, d_model = embedding.size()
        intensity_embedding = self.Embedding(intensity)

        target = target[:, :-1]  # target ????????????
        if isinstance(test_mode_target, torch.Tensor):
            target_embedding = torch.zeros((bs, src_len - 1, d_model)).to(embedding.device)
            for i in range(test_mode_target.size(0)):
                for j in range(test_mode_target.size(1)):
                    # target_embedding[i, j, :] = get_onehot_label(test_mode_target[i, j], self.d_model)
                    target_embedding[i, j, :] = self.Embedding(test_mode_target[i, j])
        else:
            # target_embedding = get_onehot_label(target, self.d_model)  # bs x src-1 x d_model
            target_embedding = self.Embedding(target)
        embedding, intensity_embedding, target_embedding, target_key_embedding, pos_embed, target_pos_embed = self.pack(
            embedding, intensity_embedding, target, target_embedding)
        embedding, intensity_embedding, target_embedding, target_key_embedding, pos_embed, target_pos_embed = self.permute(
            embedding, intensity_embedding, target_embedding, target_key_embedding, pos_embed, target_pos_embed)
        spb, bs, hidden_dim = embedding.size()
        tgt_mask = torch.triu(torch.ones(spb, spb).to(embedding.device).to(torch.bool), 1)
        src_mask = torch.triu(torch.ones(spb, spb).to(embedding.device).to(torch.bool), 1)
        memory_mask = torch.triu(torch.ones(spb, spb).to(embedding.device).to(torch.bool), 1)

        output = self.transformer(encoder_src=embedding,
                                  intensity_src=intensity_embedding,
                                  decoder_src=target_key_embedding,
                                  decoder_key=target_key_embedding,
                                  pos_embed=pos_embed,
                                  target_pos_embed=target_pos_embed,
                                  target_key_pos_embed=target_pos_embed,
                                  tgt_mask=tgt_mask,
                                  src_mask=src_mask,
                                  memory_mask=memory_mask)  # decoder_number, *,*,*
        # ?????????decoder_number, bs, src_len, d_model, ???????????????decoder??????
        output = output[-1]
        # output = output.permute(1, 0, 2)
        outputs_class = self.class_embed(output)
        return outputs_class


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None):
        super().__init__()
        self.src_len = args.src_len
        self.pos_emb_type = args.pos_emb_type
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self.fc = nn.Linear(2 * d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, encoder_src,
                intensity_src: Optional[Tensor] = None,  # ???????????????embedding
                decoder_src: Optional[Tensor] = None,  # attn_mask ????????????????????? ???detr??????None
                decoder_key: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None,
                target_pos_embed: Optional[Tensor] = None,
                target_key_pos_embed: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,  # decoder self_attn mask
                src_mask: Optional[Tensor] = None,  # encoder self_attn mask
                memory_mask: Optional[Tensor] = None,  # decoder-encoder attn mask
                ):

        # flatten NxCxHxW to HWxNxC
        spb, bs, hd = encoder_src.shape

        memory = self.encoder(src=encoder_src, pos=pos_embed)
        intensity = self.encoder(src=intensity_src, pos=pos_embed)
        memory = self.fc(torch.cat((memory, intensity), dim=-1))

        # return memory
        # print(memory.size())
        tgt = decoder_src
        # print(tgt.size(),memory.size(),tgt_mask.size(),decoder_key.size())

        # print(tgt.size())
        '''
        hs = self.decoder(tgt, memory, tgt_mask=tgt_mask,memory_mask=memory_mask,query_pos=target_key_pos_embed,
                        tgt_key=decoder_key,tgt_value=decoder_key,
                          key_pos=target_key_pos_embed, memory_pos=pos_embed)
        '''
        hs = self.decoder(tgt, memory,
                          tgt_mask=tgt_mask,
                          tgt_key=decoder_key,
                          tgt_value=decoder_key,
                          memory_mask=None,
                          memory_pos=pos_embed,
                          key_pos=target_key_pos_embed,
                          query_pos=target_key_pos_embed)
        ##print("4.3decoder??????",hs.size())
        # return memory
        return hs.transpose(1, 2), memory.permute(1, 0, 2)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_key: Optional[Tensor] = None,
                tgt_value: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory,
                           tgt_key=tgt_key,
                           tgt_mask=tgt_mask,
                           tgt_value=tgt_value,
                           memory_mask=memory_mask,
                           memory_pos=memory_pos,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           key_pos=key_pos,
                           query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # v = self.with_pos_embed(src, pos)
        v = src
        src2 = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        # ???????????????v????????????pos_embed???????????????????????????
        # v = self.with_pos_embed(src2, pos)
        v = src2
        src2 = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_key: Optional[Tensor] = None,
                     tgt_value: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     memory_pos: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     key_pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt_q = self.with_pos_embed(tgt, query_pos)
        tgt_k = tgt_q if tgt_key == None else self.with_pos_embed(tgt_key, key_pos)
        # tgt_v = tgt if tgt_value == None else self.with_pos_embed(tgt_value, key_pos)  # ??????pos_embed
        tgt_v = tgt if tgt_value == None else tgt_value  # ??????pos_embed
        tgt2 = self.self_attn(tgt_q, tgt_k, value=tgt_v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=self.with_pos_embed(memory, memory_pos), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_key: Optional[Tensor] = None,
                    tgt_value: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    memory_pos: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    key_pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt = self.norm1(tgt)
        tgt_key = self.norm1(tgt_key)

        tgt_q = self.with_pos_embed(tgt, query_pos)
        tgt_k = tgt_q if tgt_key == None else self.with_pos_embed(tgt_key, key_pos)
        # tgt_v = tgt if tgt_value==None else tgt_value
        # tgt_v = self.with_pos_embed(tgt_key, key_pos)  # ?????????pos_embed
        tgt_v = tgt_key  # ?????????pos_embed
        tgt2 = self.self_attn(tgt_q, tgt_k, value=tgt_v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_key: Optional[Tensor] = None,
                tgt_value: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt=tgt, memory=memory,
                                    tgt_key=tgt_key, tgt_value=tgt_value, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask, memory_pos=memory_pos,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask, key_pos=key_pos,
                                    query_pos=query_pos)
        return self.forward_post(tgt=tgt, memory=memory,
                                 tgt_key=tgt_key, tgt_value=tgt_value, tgt_mask=tgt_mask,
                                 memory_mask=memory_mask, memory_pos=memory_pos,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask, key_pos=key_pos, query_pos=query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        args=args
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# def get_onehot_label(x, d_model):
#     # x = torch.IntTensor(x)
#     out = torch.zeros(x.size(), d_model).to(x.device)
#     index = x.unsqueeze(1)
#     out = out.scatter_(1, index.long(), 1)
#     return out
def get_onehot_label(x, d_model):
    # x = torch.IntTensor(x)
    size = x.size() + (d_model,)
    out = torch.zeros(size).to(x.device)
    index = x.unsqueeze(-1)
    out = out.scatter_(-1, index.long(), 1)
    return out


class MyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(MyCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # logits: [BS, len, C], target: [BS, 1]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            # logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW???1]
        logits_class = torch.argmax(logits, 1)
        ones = torch.ones_like(target)
        punishment = (logits_class - target.squeeze()).pow(2).unsqueeze(1)
        punishment = torch.where(punishment < 0.5, ones, punishment)
        logits = F.log_softmax(logits, 1)
        logits = logits.gather(1, target)  # [NHW, 1]
        # print('logits:',logits)
        # print('punishment:',punishment)
        logits = punishment * logits
        # print('logit*punishment:', logits)
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # N,C,H,W => N,C,H*W
            # logits = logits.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1, logits.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(logits)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
