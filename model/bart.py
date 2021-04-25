from abc import ABC

from transformers.models.bart.modeling_bart import BartModel, BartConfig, BartEncoder, BartDecoder, PretrainedBartModel, \
    BaseModelOutputWithPastAndCrossAttentions, Seq2SeqModelOutput, BaseModelOutput, shift_tokens_right, _expand_mask, \
    logger
from torch import nn, Tensor
import torch
from model.resnet import resnet50, ResNet, Bottleneck
from typing import Optional, Tuple, Any
import argparse
import torch.nn.functional as F
import random
from typing import Dict, List, Optional, Tuple


class MyBart(BartModel):

    def __init__(self, config: BartConfig, args):
        super(BartModel, self).__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size

        self.shared = nn.Embedding(num_embeddings=args.class_number, embedding_dim=config.d_model)
        self.resnet = MyResnet(config)

        # self.label2vec = nn.Embedding(num_embeddings=args.class_number, embedding_dim=config.d_model)
        self.encoder = MyEncoder(config, self.resnet)
        self.intensity_encoder = MyEncoder(config, self.shared)
        self.decoder = MyDecoder(config, self.shared)
        self.args = args
        self.linear = nn.Linear(config.d_model, args.class_number)
        self.linear2 = nn.Linear(2 * config.d_model, config.d_model)
        self.batchnorm = nn.BatchNorm1d(4)
        self.init_weights()
        pthfile = args.pretrained_dir
        state_dict = torch.load(pthfile)
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self.resnet.load_state_dict(state_dict, False)

    def tie_weights(self):
        return

    def forward(
            self,
            input_ids=None,
            input_intensity_labels=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # decoder_input设为了一个序列长度为1的全零向量，所以这一步应该也不会执行
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            # input_ids = input_ids
        elif inputs_embeds is not None:
            inputs_embeds = self.resnet(inputs_embeds)
            # inputs_embeds = self.batchnorm(inputs_embeds)
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if input_intensity_labels is not None:
            inputs_intensity_embeds = self.shared(input_intensity_labels)
            # inputs_intensity_embeds = self.batchnorm(inputs_intensity_embeds)
            inputs_embeds = torch.cat((inputs_embeds, inputs_intensity_embeds), dim=-1)
            inputs_embeds = self.linear2(inputs_embeds)
        attention_mask = torch.ones(input_shape[0], input_shape[1]).to(self.args.device)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )


        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        # config中将return dict设置为了false，所以这一步应该不会执行
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = ()
        for i in range(4):
            if decoder_input_ids is not None:
                dec_size = decoder_input_ids.size()[:2]
                decoder_attention_mask = torch.ones(dec_size[0], dec_size[1]).to(self.args.device)
            elif decoder_inputs_embeds is not None:
                dec_size = decoder_inputs_embeds.size()[:2]
                decoder_attention_mask = torch.ones(dec_size[0], dec_size[1]).to(self.args.device)
            else:
                decoder_attention_mask = None

            # make masks if user doesn't supply
            # if not use_cache:
            #     if decoder_input_ids is not None:
            #         decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
            #             self.config,
            #             input_ids,
            #             decoder_input_ids=decoder_input_ids,
            #             decoder_padding_mask=decoder_attention_mask,
            #             causal_mask_dtype=self.shared.weight.dtype,
            #         )
            #     else:
            #         decoder_input_embeds, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
            #             self.config,
            #             input_ids,
            #             decoder_input_ids=decoder_inputs_embeds,
            #             decoder_padding_mask=decoder_attention_mask,
            #             causal_mask_dtype=self.shared.weight.dtype,
            #         )
            # else:
            #     decoder_padding_mask, causal_mask = None, None

            assert decoder_input_ids is not None or decoder_inputs_embeds is not None

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                encoder_head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            decoder_inputs_embeds = torch.cat((decoder_inputs_embeds, decoder_outputs[0][:, -1:, :]), dim=1)
            decoder_input_ids = None

        # return
        if not return_dict:
            out = decoder_outputs + encoder_outputs
            out = (self.linear(out[0]),) + out
        else:
            out = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        return out


class MyEncoder(BartEncoder):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MyDecoder(BartDecoder):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            encoder_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class MyResnet(ResNet):
    def __init__(self, config: BartConfig):
        super().__init__(Bottleneck, [3, 4, 6, 3], num_classes=config.d_model)
        self.embedding_dim = config.d_model
        self.padding_idx = None
        self.weight = torch.tensor(1.0)

    def forward(self, x: Tensor) -> Tensor:
        size = x.size()
        # print('size of imgs', size)
        x = x.reshape(size[0] * size[1], size[2], size[3], size[4])
        x = super().forward(x)
        size = size[:2] + (self.embedding_dim,)
        x = x.reshape(*size)
        return x


class MyMseLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_matirx = self.matrix().to(args.device)
        self.device = args.device

    def forward(self, preds, labels):
        # preds  (bs x len) x n_class
        # labels  (bs x len) x 1
        preds = torch.softmax(preds, dim=-1)
        loss = torch.tensor(0.0, requires_grad=True).to(self.device)
        size = preds.size()
        for i in range(size[0]):
            loss = loss + torch.mm(preds[i].unsqueeze(0), self.loss_matirx[labels[i]].unsqueeze(-1))
        return loss / size[0]

    def matrix(self):
        loss_matrix = torch.empty((38, 38))
        for i in range(0, 38):
            for j in range(0, 38):
                loss_matrix[i, j] = (i - j) * (i - j)
        return loss_matrix


if __name__ == '__main__':
    args = argparse()
    config = BartConfig(return_dict=False,
                        max_position_embeddings=4,
                        d_model=512,
                        encoder_attention_heads=8,
                        decoder_attention_heads=8,
                        encoder_layers=1,
                        decoder_layers=1)

    mybart = MyBart(config, args)
