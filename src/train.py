import torch
from typing import Tuple


def forward(input_batch, loss, batch_size, feature_extractor, encoder, decoder, criterion, one_hot_encoding, device,
            use_multi_gpu: bool) -> Tuple[torch.Tensor, float]:

    feature_extractor.train()
    encoder.train()
    decoder.train()

    batch_size = input_batch['video'].size(0)
    max_sequence_length = input_batch['video'].size(1)

    # reshape/concat the videos from the batch for alex net input
    # TODO: implement collate method in dataloader
    #? https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/13

    # TODO: refactor to proper seq2seq model class
    # TODO: handle 5-dim videos
    model_name = feature_extractor.__class__.__name__
    if model_name == 'AlexNet':
        if batch_size == 1:
            inputs = torch.flatten(input_batch['video'], end_dim=1).to(device, non_blocking=True)  # flatten just batches dim=0
            encoder_input_length = torch.tensor([inputs.size(0)]).long()
            max_target_length = len(input_batch['translation'])
            combined_target_length = max_target_length
        else:
            inputs = input_batch['video'][input_batch['pad_video_mask']].to(device, non_blocking=True)
            encoder_input_length = input_batch['pad_video_mask'].sum(dim=1)
            max_target_length = input_batch['pad_translation_mask'].sum(dim=1).max().item()
            combined_target_length = input_batch['pad_translation_mask'].sum().item()
    else:
        # TODO: extract window_size
        window_size = 16
        if batch_size == 1:
            inputs = torch.flatten(input_batch['video'], end_dim=1).to(device, non_blocking=True)
            encoder_input_length = torch.tensor([inputs.size(0)]).long()

            # resize the inputs to have a depth dimension - (seq_len, c, window_size, h, w)
            c, h, w = list(inputs.size()[1:])
            # depth_chunk_size = encoder_input_length.item() // window_size
            # window_mod = encoder_input_length.item() % window_size
            depth_chunk_size = encoder_input_length // window_size
            window_mod = encoder_input_length % window_size

            # if the frames are not dividable by the window size, append zero frames
            if window_mod > 0:
                depth_chunk_size += 1
                fill_size = window_size - window_mod
                fill_tensor = inputs.new_zeros((fill_size, c, h, w))

                inputs = torch.cat((inputs, fill_tensor))

            inputs = inputs.contiguous().view(depth_chunk_size, c, window_size, w, h)

            max_target_length = len(input_batch['translation'])
            combined_target_length = max_target_length
        else:
            encoder_input_length = input_batch['pad_video_mask'].sum(dim=1)
            max_encoder_input_length = max(encoder_input_length).unsqueeze(0)

            c, h, w = list(input_batch['video'].size()[2:])
            window_mod = max_encoder_input_length % window_size
            # if the max input is dividable by the window size go for it
            # otherwise add one extra depth chunk
            depth_chunk_size = max_encoder_input_length // window_size if window_mod == 0 else max_encoder_input_length // window_size + 1

            # special case: if not enough pre-padded values are existing
            if input_batch['video'].size(1) < depth_chunk_size * window_size:
                # pad extra zeros to the batch
                fill_size = window_size - window_mod
                video_fill_tensor = input_batch['video'].new_zeros((batch_size, fill_size, c, h, w))
                pad_fill_tensor = input_batch['pad_video_mask'].new_zeros((batch_size, fill_size))

                input_batch['video'] = torch.cat((input_batch['video'], video_fill_tensor), dim=1)
                input_batch['pad_video_mask'] = torch.cat((input_batch['pad_video_mask'], pad_fill_tensor), dim=1)

            pad_mask = torch.zeros_like(input_batch['pad_video_mask'])
            pad_mask[:, :depth_chunk_size * window_size] = True

            # resize inputs
            inputs = input_batch['video'][pad_mask].to(device, non_blocking=True)
            inputs = inputs.contiguous().view(depth_chunk_size * batch_size, window_size, c, h, w).permute(0, 2, 1, 3, 4)

            max_target_length = input_batch['pad_translation_mask'].sum(dim=1).max().item()
            combined_target_length = input_batch['pad_translation_mask'].sum().item()
        encoder_input_length = depth_chunk_size.repeat(batch_size)

    # get one-hot encoding for each word in each sentence
    targets = torch.tensor([tuple([one_hot_encoding.word2index[word] for word in word_tuple]) for word_tuple in input_batch['translation']], dtype=torch.int64)

    targets = targets.to(device, non_blocking=True)

    ### ENCODER ###
    if torch.cuda.device_count() > 1 and use_multi_gpu:
        hidden = encoder.module.initHidden(batch_size, device)
        input_size = encoder.module.input_size
    else:
        hidden = encoder.initHidden(batch_size, device)
        input_size = encoder.input_size

    ### FEATURE EXTRACTOR ###
    inputs = inputs.contiguous()

    # TODO: make feature extractor trainable based on input args
    with torch.no_grad():
        feature_extractor_out = feature_extractor(inputs)

    # create zero padded sequence and fill in feature extractor output
    if model_name == 'AlexNet' and batch_size != 1:
        zero_seq = feature_extractor_out.new_full((batch_size * max_sequence_length, input_size), fill_value=0.0)
        zero_seq[input_batch['pad_video_mask'].view(-1)] = feature_extractor_out
        feature_extractor_out = zero_seq

    ### ENCODER ###
    # reshape feature extractor output to batch level
    # first reshape whole sequence to batches, then permute to (seq_len, batch_size, ..)
    enc_in = feature_extractor_out.view(batch_size, -1, input_size).permute(1, 0, 2)

    max_batch_sequence_length = encoder_input_length.max().item()

    # unsequeeze for dataparallelism (1, batch_size) -> will be scattered in batch_size dim
    encoder_input_length = encoder_input_length.unsqueeze(0)

    enc_out, enc_hidden = encoder(enc_in, hidden, max_batch_sequence_length, encoder_input_length)
    # enc_out, enc_hidden = encoder(enc_in, hidden)

    ### DECODER ###
    dec_in_words = ['<SOS>'] * batch_size
    dec_hidden = enc_hidden

    # get onehot encoding foreach dec_in word. as tensor with size (seq_len, batch_size, encoding_size)
    dec_in = torch.cat([one_hot_encoding(word) for word in dec_in_words]).unsqueeze(0).type(torch.float32)  # unsqueeze for seq_len=1
    dec_in = dec_in.to(device)

    # create attention padding mask
    pad_mask = torch.arange(max_batch_sequence_length)[None, :] < encoder_input_length.permute(1, 0)

    for target_idx in range(max_target_length):

        dec_out, dec_hidden = decoder(dec_in, dec_hidden, enc_out, pad_mask)
        loss += criterion(dec_out.squeeze(0), targets[target_idx])

        # get top word
        _, topi = dec_out.topk(1)

        # create new input for next iteration - onehot vector based on topi word idx from previous decoder output
        dec_in = dec_out.clone().detach()

        for i, top_idx in enumerate(topi[0]):
            dec_in[:, i, top_idx] = 1
        dec_in[dec_in < 1] = 0

    return loss, loss.item() / combined_target_length


def backward(loss, encoder, decoder, encoder_optimizer, decoder_optimizer, clip_gradients, max_target_length) -> None:
    loss.backward()

    ### Gradient Clipping ###
    # helps prevent the exploding gradient problem in RNNs
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_gradients)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_gradients)

    encoder_optimizer.step()
    decoder_optimizer.step()
