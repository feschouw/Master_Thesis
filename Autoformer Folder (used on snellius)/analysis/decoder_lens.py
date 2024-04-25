import torch


def encoder_forward_decoderlens(encoder, x, attn_mask=None):
    tmp_xs = []  # added by Angela
    attns = []
    if encoder.conv_layers is not None:
        # never here(?)
        for attn_layer, conv_layer in zip(encoder.attn_layers, encoder.conv_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask)
            x = conv_layer(x)
            attns.append(attn)
        x, attn = encoder.attn_layers[-1](x)
        attns.append(attn)
    else:
        for attn_layer in encoder.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
            tmp_xs.append(x)  # added by Angela
            # You might think it is necessary to make a deepcopy of x before appending it to the list,
            # but this is not necessary, as x is a new instance every loop

    if encoder.norm is not None:
        x = encoder.norm(x)

    return x, attns, tmp_xs


def model_forward_decoderlens(
    model,
    x_enc,
    x_mark_enc,
    x_dec,
    x_mark_dec,
    enc_model_mask=None,
    dec_model_mask=None,
    dec_enc_mask=None,
    tmp_layernorm=True,
):

    # decomp init
    mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, model.pred_len, 1)
    zeros = torch.zeros(
        [x_dec.shape[0], model.pred_len, x_dec.shape[2]], device=x_enc.device
    )
    seasonal_init, trend_init = model.decomp(x_enc)
    # decoder input
    trend_init = torch.cat([trend_init[:, -model.label_len :, :], mean], dim=1)
    seasonal_init = torch.cat([seasonal_init[:, -model.label_len :, :], zeros], dim=1)
    # enc
    enc_out = model.enc_embedding(x_enc, x_mark_enc)
    # enc_out, attns = model.encoder(enc_out, attn_mask=enc_model_mask)
    enc_out, attns, tmp_xs = encoder_forward_decoderlens(
        model.encoder, enc_out, attn_mask=enc_model_mask
    )
    # dec
    dec_out = model.dec_embedding(seasonal_init, x_mark_dec)
    seasonal_part, trend_part = model.decoder(
        dec_out,
        enc_out,
        x_mask=dec_model_mask,
        cross_mask=dec_enc_mask,
        trend=trend_init,
    )
    # final
    dec_out = trend_part + seasonal_part

    tmp_out = []
    for tmp_x in tmp_xs:
        if tmp_layernorm:  # this should always happen for true decoder lens
            tmp_x = model.encoder.norm(tmp_x)
        tmp_dec_out = model.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = model.decoder(
            tmp_dec_out,
            tmp_x,
            x_mask=dec_model_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )
        # final
        tmp_dec_out = trend_part + seasonal_part
        tmp_out.append(tmp_dec_out[:, -model.pred_len :, :])

    if model.output_attention:
        return dec_out[:, -model.pred_len :, :], attns, tmp_out
    else:
        return dec_out[:, -model.pred_len :, :], tmp_out  # [B, L, D]
