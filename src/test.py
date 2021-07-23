import logging

import torch

from evaluation.bleu import moses_multi_bleu as bleu
from utils.time import startTime, timeSince


def evaluate(alexnet,
             encoder,
             decoder,
             dataset_generator,
             one_hot_encoding,
             device,
             beam_width: int,
             use_multi_gpu: bool,
             flip_sequence: bool,
             ignore_tokens: list = ['<SOS>', '<EOS>', '<UNK>']) -> None:

    logger = logger = logging.getLogger('SignLanguageTranslation')

    batch_size = dataset_generator.batch_size

    total_bleu_score = 0.

    eval_batch_count = len(dataset_generator)
    start = startTime()

    with torch.no_grad():
        # set evaluation mode for models
        alexnet.eval()
        encoder.eval()
        decoder.eval()

        for i_batch, sample_batched in enumerate(dataset_generator):

            if flip_sequence:
                sample_batched['video'] = torch.flip(sample_batched['video'], dims=[1])

            batch_size = sample_batched['video'].size(0)
            max_sequence_length = sample_batched['video'].size(1)

            # reshape/concat the videos from the batch for alex net input
            if batch_size == 1:
                inputs = torch.flatten(sample_batched['video'], end_dim=1).to(device, non_blocking=True)  # flatten just batches dim=0
                encoder_input_length = torch.tensor([inputs.size(0)]).long()
                max_target_length = len(sample_batched['translation'])
            else:
                inputs = sample_batched['video'][sample_batched['pad_video_mask']].to(device, non_blocking=True)
                encoder_input_length = sample_batched['pad_video_mask'].sum(dim=1)
                max_target_length = sample_batched['pad_translation_mask'].sum(dim=1).max().item()

            # get one-hot encoding for each word in each sentence
            targets = torch.tensor([tuple([one_hot_encoding.word2index[word] for word in word_tuple]) for word_tuple in sample_batched['translation']],
                                   dtype=torch.int64)

            inputs = inputs.to(device)
            targets = targets.to(device)

            ### CNN ###
            alex_out = alexnet(inputs)

            ### ENCODER ###
            if torch.cuda.device_count() > 1 and use_multi_gpu:
                hidden = encoder.module.initHidden(batch_size, device)
                input_size = encoder.module.input_size
            else:
                hidden = encoder.initHidden(batch_size, device)
                input_size = encoder.input_size

            # create zero padded sequence and fill in alexnet output
            if batch_size != 1:
                zero_seq = alex_out.new_full((batch_size * max_sequence_length, input_size), fill_value=0.0)
                zero_seq[sample_batched['pad_video_mask'].view(-1)] = alex_out
                alex_out = zero_seq

            # reshape alex net output to batch level
            # first reshape whole sequence to batches, then permute to (seq_len, batch_size, ..)
            enc_in = alex_out.view(batch_size, -1, input_size).permute(1, 0, 2)

            max_batch_sequence_length = encoder_input_length.max().item()

            # unsequeeze for dataparallelism (1, batch_size) -> will be scattered in batch_size dim
            encoder_input_length = encoder_input_length.unsqueeze(0)

            enc_out, enc_hidden = encoder(enc_in, hidden, max_batch_sequence_length, encoder_input_length)
            # enc_out, enc_hidden = encoder(enc_in, hidden)

            ### DECODER ###
            dec_translation = [[] for _ in range(batch_size)]
            dec_in_words = ['<SOS>'] * batch_size
            dec_hidden = enc_hidden

            # get onehot encoding foreach dec_in word. as tensor with size (seq_len, batch_size, encoding_size)
            dec_in = torch.cat([one_hot_encoding(word) for word in dec_in_words]).unsqueeze(0).type(torch.float32)  # unsqueeze for seq_len=1
            dec_in = dec_in.to(device)

            # create attention padding mask
            pad_mask = torch.arange(max_batch_sequence_length)[None, :] < encoder_input_length.permute(1, 0)

            ## First decoder run not looped - getting to beam_width * batch_size
            dec_out, dec_hidden = decoder(dec_in, dec_hidden, enc_out, pad_mask)

            # get top beam words for each batch
            topk, topi = dec_out.topk(beam_width)

            # compute log for all topk
            topk = torch.log(topk)

            # create new input for next iteration - onehot vector based on topi word idx from previous decoder output
            dec_in = dec_out.new_zeros((1, beam_width * batch_size, dec_out.size(-1)))

            # flatten topi, take each index and fill ones for each index
            for i, idx in enumerate(topi[0].contiguous().view(-1)):
                dec_in[:, i, idx] = 1

            # get words from encoder indexes and their corresponding scores
            dec_out_words = [[(one_hot_encoding.index2word[enc_idx.item()], score.item()) for enc_idx, score in zip(batch[0], batch[1])]
                             for batch in zip(topi[0], topk[0])]

            # add predicted words to translations
            for batch_idx in range(batch_size):
                dec_translation[batch_idx].append(dec_out_words[batch_idx])

            # repeat hidden, enc_out and pad mask to beam_width-times batches - reapeat_interleave will repeat inplace
            # [batch_1, value]               [batch_1, value]
            # [batch_1, value]               [batch_2, value]
            # [batch_2, value]  rather than  [batch_2, value]
            # [batch_2, value]               [batch_1, value]
            pad_mask = torch.repeat_interleave(pad_mask, beam_width, dim=0)
            enc_out = torch.repeat_interleave(enc_out, beam_width, dim=1)
            dec_hidden = torch.repeat_interleave(dec_hidden, beam_width, dim=1)

            # TODO: make use of a dictionary?
            # create beam list - [((original_batch, previous_batch (here original_batch), cur_batch_idx, score, encoding_idx))]
            beams = [[batch_idx // beam_width, batch_idx // beam_width, batch_idx % beam_width,
                      score.item(), topi[0].contiguous().view(-1)[batch_idx].item()] for batch_idx, score in enumerate(topk[0].contiguous().view(-1))]

            # loop over max target length - 1, since we already computed one step
            for idx in range(max_target_length - 1):

                dec_out, dec_hidden = decoder(dec_in, dec_hidden, enc_out, pad_mask)

                # find the best beam_width outputs for each batch (beam_width * batch_size)
                topk, topi = dec_out.topk(beam_width, sorted=False)

                # compute log for all topk
                topk = torch.log(topk)

                # beam_count = number of original batches * beam_width * beam_width (nodes per batch step)
                beam_count = beam_width * batch_size * beam_width  # || topk[0].view(-1).size(0)

                new_beams = []
                # compute scores for each beam node (path)
                for idx, (top_score, top_idx) in enumerate(zip(topk[0].contiguous().view(-1), topi[0].contiguous().view(-1))):
                    # split flattened tensor back to batch_size indexes
                    orig_batch_idx = idx // (beam_count // batch_size)

                    # get the batch index which the value was computed on
                    # this only works if topi and topk are contiguous flattened!
                    last_batch_idx = idx // beam_width if idx < beam_count // batch_size else idx % (beam_count // batch_size) // beam_width

                    # increase current idx for each example; reset if new batch idx
                    current_batch_idx = idx % (beam_count // batch_size) if idx >= beam_count // batch_size else idx

                    # the topi tensor was flattened before, therefore, we need to recompute the batch idx and subtract the space * batches if the idx is over the space
                    top_idx = top_idx if top_idx < dec_out.size(-1) else top_idx - (top_idx // dec_out.size(-1) * dec_out.size(-1))

                    last_score = beams[last_batch_idx + orig_batch_idx * beam_width][3]
                    # # filter by original batch and current_batch (which is the previous for this iteration) -> previous_batch is the previous from the last iteration
                    # last_score = list(filter(lambda x: x[0] == orig_batch_idx and x[2] == last_batch_idx in x, beams))[0][3]

                    # compute score and add beam to new beams list
                    new_beams.append([orig_batch_idx, last_batch_idx, current_batch_idx, top_score.item() + last_score, top_idx.item()])
                    #  top_score.item() + last_score, topi[0].contiguous().view(-1)[idx].item()])
                beams = new_beams

                # sort beams by original batch and score
                beams.sort(key=lambda x: (x[0], -x[3]))

                # TODO: remove beams with idx > beam_width?

                # create new input for next iteration - onehot vector based on topi word idx from previous decoder output
                dec_in = dec_in.new_zeros((1, beam_width * batch_size, dec_out.size(-1)))

                # create new decoder hidden tensor with correct values (for each best beam the corresponding hidden)
                new_dec_hidden = dec_hidden.clone()

                # fill ones in dec_in based on sorted beams (best beam items)
                # collect translation words for each best beam
                dec_out_words = [[] for _ in range(batch_size)]
                for top_beam in range(beam_width):
                    for batch_idx in range(batch_size):
                        # get beam - the previous batch_idx (current from last iteration) will help with using the correct dec_hidden value
                        # [((original_batch, previous_batch, cur_batch_idx, score, encoding_idx))]
                        orig_batch_idx, previous_batch_idx, _, score, encoding_idx = beams[top_beam + (batch_idx * (beam_count // batch_size))]

                        new_batch_idx = top_beam + batch_idx * beam_width

                        # set encoding idx to one for each batch and each top beam
                        dec_in[:, new_batch_idx, encoding_idx] = 1

                        # replace hidden with corresponding hidden values for each best beam
                        # multiply original batch idx with beam width and add previous to get the correct hidden value -> for each original batch the numbering starts with 0
                        new_dec_hidden[:, new_batch_idx, :] = dec_hidden[:, orig_batch_idx * beam_width + previous_batch_idx, :]

                        # append string representation of encoding idx to dec out words
                        dec_out_words[batch_idx].append((one_hot_encoding.index2word[encoding_idx], score))

                # add predicted words to translations
                for batch_idx in range(batch_size):
                    dec_translation[batch_idx].append(dec_out_words[batch_idx])

                # create new hidden vector for next decoding step
                # Note: enc_out and padding mask stays the same - they allready are repeated to batch_size * beam_width
                dec_hidden = new_dec_hidden

                # TODO: early stop decoding if <EOF> is reached for each sentence in each batch
                # translation is cut of after EOS anyway, so this would only increase computation time

            # take only best translation for each batch (out of beams) and filter tokens
            translation_sentences = []
            for batch_idx in range(batch_size):
                batch_sentences = []
                for sentence_idx in range(beam_width):
                    # cut off sentence after <EOS> token and remove <SOS> and <PAD>
                    sentence = []
                    score = 0
                    word_count = 0
                    for dec_step in dec_translation[batch_idx]:
                        _word, _score = dec_step[sentence_idx]
                        if _word == '<EOS>':
                            break
                        elif _word not in ignore_tokens:
                            sentence.append(_word)
                            score = _score
                            word_count += 1
                    # append sentence and normalized probability for each sentence
                    batch_sentences.append((' '.join(sentence), score / word_count))

                # append best sentence in each original batch to the translations
                translation_sentences.append(sorted(batch_sentences, key=lambda x: x[1])[0][0])

            ### EVALUATION ###
            target_sentences = []
            for i in range(len(sample_batched['translation'][0])):
                sentence_words = []
                for j in range(max_target_length):
                    if (sample_batched['translation'][j][i] not in ignore_tokens):
                        sentence_words.append(sample_batched['translation'][j][i])
                target_sentences.append(' '.join(sentence_words))

            # bleu score
            batch_bleu_score = bleu(translation_sentences, target_sentences)
            total_bleu_score += batch_bleu_score

            current_percentage = (i_batch + 1) / eval_batch_count
            logger.info(f"Batch: {i_batch+1}/{eval_batch_count}\tBatch Bleu Score: {batch_bleu_score:.4f}\t{timeSince(start, current_percentage)}")

    logger.info(f"Final-Evaluation Bleu Score on Dev-Set: {total_bleu_score / eval_batch_count}")
