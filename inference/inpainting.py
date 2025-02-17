import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))

import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from models.soundstream_hubert_new import SoundStream
from stage2 import Stage2Processor, AudioProcessor, extract_uuid
import uuid
import re
import soundfile as sf

### Custom Logits Processors ###
class VocalTokenProcessor(LogitsProcessor):
    """Ensures we only generate tokens for the vocal codebook"""
    def __init__(self, codebook_size=1024, global_offset=45334):
        super().__init__()
        self.start_id = global_offset  # Start of vocal codebook (codebook #0)
        self.end_id = global_offset + codebook_size  # End of vocal codebook
        
    def __call__(self, input_ids, scores):
        scores[:, :self.start_id] = -float('inf')
        scores[:, self.end_id:] = -float('inf')
        return scores

class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores

class VocalRangeProcessor(LogitsProcessor):
    """Only allows tokens in the vocal codebook range"""
    def __init__(self, start_id=45334, size=1024):
        self.start_id = start_id
        self.end_id = start_id + size
        
    def __call__(self, input_ids, scores):
        valid_scores = scores[:, self.start_id:self.end_id].clone()
        scores[:, :] = float('-inf')
        scores[:, self.start_id:self.end_id] = valid_scores
        return scores

### Audio and Lyrics Utilities ###
def encode_input_audio(audio_path, codec_model, semantic_model, device):
    """Encode MusicGen audio using both acoustic and semantic paths"""
    print("\nProcessing input audio...")
    audio, sr = torchaudio.load(audio_path)
    print(f"Loaded audio: shape={audio.shape}, sr={sr}")
    
    # Convert to mono and resample if needed
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
    
    audio = audio.unsqueeze(0).to(device)  # [1, 1, samples]
    print(f"Preprocessed audio shape: {audio.shape}")
    
    with torch.no_grad():
        # Get semantic features
        x = audio[:, 0, :]
        x = F.pad(x, (160, 160))
        semantic_output = semantic_model(x, output_hidden_states=True).hidden_states
        semantic_features = torch.stack(semantic_output, dim=1).mean(1)
        print(f"Semantic features shape: {semantic_features.shape}")
        
        # Get acoustic features
        e_semantic = codec_model.encoder_semantic(semantic_features.transpose(1, 2))
        e_acoustic = codec_model.encoder(audio)
        print(f"Acoustic features shape: {e_acoustic.shape}")
        
        if e_acoustic.shape[2] != e_semantic.shape[2]:
            print("Adjusting acoustic features shape...")
            x_pad = F.pad(audio[:, 0, :], (160, 160)).unsqueeze(0)
            e_acoustic = codec_model.encoder(torch.transpose(x_pad, 0, 1))
        
        print(f"Before concat - Acoustic: {e_acoustic.shape}, Semantic: {e_semantic.shape}")
        e = torch.cat([e_acoustic, e_semantic], dim=1)
        print(f"After concat: {e.shape}")
        e = codec_model.fc_prior(e.transpose(1, 2)).transpose(1, 2)
        print(f"After fc_prior: {e.shape}")
        
        bw = codec_model.target_bandwidths[0] if hasattr(codec_model, 'target_bandwidths') and codec_model.target_bandwidths else 0.5
        quantized, codes, bandwidth, _ = codec_model.quantizer(e, codec_model.frame_rate, bw)
        print(f"Final codes shape: {codes.shape}")
        
        codes = codes.squeeze().cpu().numpy()
        instrument_codes = codes[1] if len(codes.shape) > 1 else codes
        print(f"Instrument codes shape: {instrument_codes.shape}")
        print(f"Code value range: min={instrument_codes.min()}, max={instrument_codes.max()}")
        
        if len(instrument_codes.shape) > 1:
            instrument_codes = instrument_codes.flatten()
        
    return instrument_codes

def split_lyrics(lyrics):
    """Split lyrics into segments based on a regex pattern."""
    pattern = r"\[(\w+)\](.*?)\n(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    return [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]

def crop_audio_to_60_seconds(audio_path, output_path=None):
    """
    Crop audio to exactly 60 seconds using soundfile for more reliable duration handling.
    """
    print("\nPreprocessing audio...")
    
    # Load with soundfile
    audio, sr = sf.read(audio_path)
    duration = len(audio)/sr
    print(f"Original audio: shape={audio.shape}, sr={sr}, duration={duration:.2f}s")
    
    target_duration = 60  # seconds
    target_samples = target_duration * sr
    
    if len(audio) < target_samples:
        print(f"Warning: Audio is shorter than {target_duration} seconds ({duration:.2f}s)")
        return audio_path
        
    # Crop to exact number of samples for 60 seconds
    audio = audio[:target_samples]
    if len(audio.shape) == 1:
        audio = audio.reshape(1, -1)  # Ensure 2D array
    else:
        audio = audio.T  # Convert to channels-first format
        
    new_duration = audio.shape[1]/sr
    print(f"Cropped audio: shape={audio.shape}, duration={new_duration:.2f}s")
    
    output_path = output_path or audio_path
    
    # Convert to torch tensor for saving
    audio_tensor = torch.FloatTensor(audio)
    torchaudio.save(output_path, audio_tensor, sr)
    print(f"Saved cropped audio to: {output_path}")
    
    # Verify the saved file
    verify_info = sf.info(output_path)
    print(f"Verified saved audio: duration={verify_info.duration:.2f}s, sr={verify_info.samplerate}")
    
    return output_path

### Generation Function with Explicit Segmentation ###
def generate_with_segments(
    model,
    mmtokenizer,
    codectool,
    genres,
    lyrics_segments,
    instrument_codes,
    device,
    run_n_segments,
    max_new_tokens=2000,
    temperature=1.0,
    top_p=0.93,
    repetition_penalty=1.2
):
    """
    Generate vocals segment by segment while enforcing instrument tokens.
    If there is more than one lyric section, the first prompt is the full context (all lyrics)
    and subsequent prompts are individual segments.
    If there's only one section, then the prompt is simply the full context.
    """
    print("\nStarting segmented generation...")

    if len(lyrics_segments) > 1:
        # Combine all lyric sections for full context.
        full_lyrics = "\n".join(lyrics_segments)
        full_context = f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"
        # Use the full context as the setup iteration, then each individual section.
        prompt_texts = [full_context] + lyrics_segments
    else:
        # Only one section: just use it.
        full_context = f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{lyrics_segments[0]}"
        prompt_texts = [full_context]

    # Standard segment markers.
    start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
    end_of_segment = mmtokenizer.tokenize('[end_of_segment]')

    # Use a logits processor to restrict outputs to the vocal codebook range.
    vocal_range = VocalRangeProcessor(start_id=45334, size=1024)

    all_tokens = []
    raw_output = None

    total_instrument_tokens = len(instrument_codes)
    # If there are multiple segments, divide tokens among them;
    # otherwise, use all instrument tokens.
    if len(lyrics_segments) > 1:
        tokens_per_segment = total_instrument_tokens // (len(lyrics_segments))
    else:
        tokens_per_segment = total_instrument_tokens
    print("\nSegment Distribution:")
    print(f"Total instrument tokens: {total_instrument_tokens}")
    print(f"Tokens per segment: {tokens_per_segment}")
    print(f"Number of segments (excluding context): {len(lyrics_segments) if len(lyrics_segments)>1 else 1}")

    # Determine how many iterations to run.
    if len(prompt_texts) > 1:
        run_n = min(run_n_segments + 1, len(prompt_texts))
    else:
        run_n = 1  # Only one prompt if there's a single section

    for i in range(run_n):
        print(f"\nProcessing iteration {i}/{run_n - 1}")
        # Remove any segment markers from the current text.
        section_text = prompt_texts[i].replace('[start_of_segment]', '').replace('[end_of_segment]', '')

        # If there are multiple prompts, use iteration 0 as the setup (full context).
        if len(prompt_texts) > 1 and i == 0:
            print("Setup iteration: saving context.")
            continue

        segment_idx = i - 1 if len(prompt_texts) > 1 else 0
        start_idx = segment_idx * tokens_per_segment
        end_idx = start_idx + tokens_per_segment if i < run_n - 1 else total_instrument_tokens
        print(f"Token range for segment {segment_idx}: {start_idx} to {end_idx}")

        # Build the prompt.
        if i == 0 or (len(prompt_texts) == 1):
            # If only one prompt is present, use it entirely.
            head_tokens = mmtokenizer.tokenize(prompt_texts[0])
            # (No need to append a duplicate stage marker)
            prompt_ids = head_tokens + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
            prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
        else:
            # For subsequent segments (when more than one prompt exists), use the context from the previous output.
            new_segment = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
            new_segment = torch.as_tensor(new_segment).unsqueeze(0).to(device)
            if raw_output is not None:
                prompt_ids = torch.cat([raw_output, new_segment], dim=1)
            else:
                prompt_ids = new_segment

        print(f"Segment {segment_idx} prompt length: {prompt_ids.shape[1]}")
        print(f"Current segment lyrics:\n{section_text}")

        current_input = prompt_ids
        segment_tokens = []

        try:
            # For each expected instrument token in the segment, generate a vocal token and force an instrument token.
            for t in tqdm(range(start_idx, end_idx), desc=f"Segment {segment_idx} Generation"):
                with torch.no_grad():
                    outputs = model(current_input)
                    logits = outputs.logits[:, -1, :]
                    logits = logits / temperature
                    logits = vocal_range(current_input, logits)
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    vocal_token = next_token[0].item()

                segment_tokens.append(vocal_token)

                # Force in the corresponding instrument token.
                inst_token = int(instrument_codes[t])
                # Adjust instrument token arithmetic as needed.
                inst_token = (inst_token % 1024) + 45334 + 1024
                segment_tokens.append(inst_token)

                # Update the context with the new pair.
                token_pair = torch.tensor([[vocal_token, inst_token]], device=device)
                current_input = torch.cat([current_input, token_pair], dim=1)

                # Maintain context window limits.
                if current_input.shape[1] > 16000:
                    current_input = current_input[:, -8000:]

                if (t - start_idx) % 100 == 0:
                    print(f"\nSegment {segment_idx} progress - Step {t - start_idx}/{end_idx - start_idx}:")
                    print(f"Vocal token: {vocal_token}")
                    print(f"Instrument token: {inst_token}")
                    print(f"Context length: {current_input.shape[1]}")

        except Exception as e:
            print(f"\nError during segment {segment_idx} generation: {str(e)}")
            if len(segment_tokens) == 0:
                raise Exception(f"No tokens generated for segment {segment_idx}")

        # Append the <EOA> token at the end of the final segment.
        if i == run_n - 1:
            segment_tokens.append(mmtokenizer.eoa)
            eoa_tensor = torch.tensor([[mmtokenizer.eoa]], device=device)
            current_input = torch.cat([current_input, eoa_tensor], dim=1)

        # Update raw_output for subsequent segments.
        if i == 1 or (len(prompt_texts) == 1):
            raw_output = current_input
        else:
            raw_output = torch.cat([raw_output, current_input[:, prompt_ids.shape[1]:]], dim=1)

        all_tokens.extend(segment_tokens)
        print(f"\nSegment {segment_idx} complete - Generated {len(segment_tokens)//2} token pairs")

    # Validate that the SOA and EOA tokens are paired correctly.
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
    print("\nGeneration Summary:")
    print(f"Total tokens generated: {len(all_tokens)}")
    print(f"SOA tokens: {len(soa_idx)}, positions: {soa_idx}")
    print(f"EOA tokens: {len(eoa_idx)}, positions: {eoa_idx}")

    if len(soa_idx) != len(eoa_idx):
        raise ValueError(f'Invalid SOA/EOA pairing: {len(soa_idx)} SOA tokens vs {len(eoa_idx)} EOA tokens')

    print("\nToken sequence validation passed!")
    return np.array(all_tokens)


def save_generated_tokens(final_tokens, output_dir, genres, codectool, generation_params):
    """Save the generated token sequences in a format compatible with Stage 2."""
    print("\nProcessing and saving generated tokens...")
    stage1_output_dir = os.path.join(output_dir, "stage1")
    os.makedirs(stage1_output_dir, exist_ok=True)
    stage1_output_set = []
    
    final_tokens = final_tokens[:2 * (len(final_tokens) // 2)]
    paired_tokens = rearrange(final_tokens, "(n b) -> b n", b=2)
    vocal_tokens = paired_tokens[0]
    instr_tokens = paired_tokens[1]
    
    print(f"Token sequence lengths - Vocal: {len(vocal_tokens)}, Instrumental: {len(instr_tokens)}")
    
    try:
        top_p = generation_params.get('top_p', 0.93)
        temperature = generation_params.get('temperature', 1.0)
        repetition_penalty = generation_params.get('repetition_penalty', 1.2)
        max_new_tokens = generation_params.get('max_new_tokens', 2000)
        random_id = uuid.uuid4()
        
        # Create codec tools with proper configuration for each stream.
        vocal_codectool = CodecManipulator("xcodec", 0, 1)
        instr_codectool = CodecManipulator("xcodec", 1, 1)
        
        vocal_codes = vocal_codectool.ids2npy(vocal_tokens)
        instr_codes = instr_codectool.ids2npy(instr_tokens)
        
        vocal_save_path = os.path.join(
            stage1_output_dir,
            f"cot_{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_vocal_{random_id}".replace('.', '@')+'.npy'
        )
        inst_save_path = os.path.join(
            stage1_output_dir,
            f"cot_{genres.replace(' ', '-')}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_instrumental_{random_id}".replace('.', '@')+'.npy'
        )
        
        np.save(vocal_save_path, vocal_codes)
        np.save(inst_save_path, instr_codes)
        
        stage1_output_set.append(vocal_save_path)
        stage1_output_set.append(inst_save_path)
        
        print(f"Saved token sequences to {stage1_output_dir}")
        return stage1_output_set
        
    except Exception as e:
        print(f"Error saving tokens: {str(e)}")
        raise

### Main function ###
def main(
    input_audio_path,
    lyrics_file,
    genre_file,
    model_path,
    stage2_model_path,
    codec_config,
    codec_checkpoint,
    output_dir,
    config_path,
    vocal_decoder_path,
    inst_decoder_path,
    run_n_segments=2,
    disable_offload_model=False,
    device="cuda"
):
    print("\nInitializing YuE Inpainting...")
    print("Loading models...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)
    model.eval()
    
    semantic_model = AutoModel.from_pretrained(
        "./xcodec_mini_infer/semantic_ckpts/hf_1_325000"
    ).to(device)
    semantic_model.eval()
    
    model_config = OmegaConf.load(codec_config)
    codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
    codec_model.load_state_dict(torch.load(codec_checkpoint, map_location='cpu')['codec_model'])
    codec_model.eval()
    
    mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
    codectool = CodecManipulator("xcodec", 0, 1)
    
    cropped_audio_path = crop_audio_to_60_seconds(
        input_audio_path,
        output_path=os.path.join(output_dir, "cropped_input.mp3")
    )
    
    instrument_codes = encode_input_audio(cropped_audio_path, codec_model, semantic_model, device)
    
    print("\nLoading genre and lyrics...")
    with open(genre_file) as f:
        genres = f.read().strip()
    with open(lyrics_file) as f:
        lyrics = f.read().strip()
    
    lyrics_segments = split_lyrics(lyrics)
    run_n_segments = min(run_n_segments + 1, len(lyrics_segments))
    lyrics_segments = lyrics_segments[:run_n_segments]
    
    generation_params = {
        'max_new_tokens': 2000,
        'temperature': 1.0,
        'top_p': 0.93,
        'repetition_penalty': 1.2,
    }
    
    final_tokens = generate_with_segments(
        model=model,
        mmtokenizer=mmtokenizer,
        codectool=codectool,
        genres=genres,
        lyrics_segments=lyrics_segments,
        instrument_codes=instrument_codes,
        device=device,
        run_n_segments=run_n_segments,
        **generation_params
    )
    
    stage1_output_set = save_generated_tokens(
        final_tokens, 
        output_dir, 
        genres, 
        codectool,
        generation_params
    )
    print("\nStage 1 complete!")
    
    if not disable_offload_model:
        model.cpu()
        del model
        torch.cuda.empty_cache()
    
    print("\nStarting Stage 2 inference...")
    stage2_processor = Stage2Processor(
        stage2_model_path=stage2_model_path,
        tokenizer_path="./mm_tokenizer_v0.2_hf/tokenizer.model",
        device=device,
        batch_size=4
    )

    stage2_output_dir = os.path.join(output_dir, "stage2")
    os.makedirs(stage2_output_dir, exist_ok=True)

    # Process stage2
    
    stage2_results = stage2_processor.process_stage2(
        stage1_output_set,
        stage2_output_dir
    )

    if not stage2_results or len(stage2_results) != 2:
        print(f"Stage 2 processing failed: expected 2 results, got {len(stage2_results)}")
        return

    # Process and save stems/mix
    recons_mix_dir = AudioProcessor.process_and_save_audio(
        stage2_results=stage2_results,
        codec_model=codec_model,
        output_dir=output_dir,
        device=device
    )

    if recons_mix_dir is None:
        print("Failed to create reconstruction mix")
        return

    # Get UUID for consistent naming
    uuid = extract_uuid(stage2_results[0])
    recons_mix_path = os.path.join(recons_mix_dir, f"mix_{uuid}.mp3")

    if not os.path.exists(recons_mix_path):
        print(f"Mix file not found at {recons_mix_path}")
        return

    AudioProcessor.apply_vocoder(
        stage2_results=stage2_results,
        config_path=config_path,
        vocal_decoder_path=vocal_decoder_path,
        inst_decoder_path=inst_decoder_path,
        output_dir=output_dir,
        codec_model=codec_model,
        recons_mix=recons_mix_path,
        rescale=True
    )
    
    print("\nInpainting complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio_path", type=str, required=True)
    parser.add_argument("--lyrics_file", type=str, required=True)
    parser.add_argument("--genre_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--stage2_model_path", type=str, required=True)
    parser.add_argument("--codec_config", type=str, required=True)
    parser.add_argument("--codec_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--vocal_decoder_path", type=str, required=True)
    parser.add_argument("--inst_decoder_path", type=str, required=True)
    parser.add_argument("--run_n_segments", type=int, default=2, help="Number of segments to process during generation.")
    parser.add_argument("--disable_offload_model", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    main(**vars(args))
