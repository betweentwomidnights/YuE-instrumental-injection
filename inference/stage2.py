import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))

import torch
import torchaudio
import numpy as np
import soundfile as sf
from tqdm import tqdm
from collections import Counter
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from codecmanipulator import CodecManipulator
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched
from mmtokenizer import _MMSentencePieceTokenizer
import copy

class BlockTokenRangeProcessor(LogitsProcessor):
    """Blocks generation of tokens outside the specified range"""
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores
    
def extract_uuid(filename):
    """Extract UUID from the original complex filename"""
    parts = os.path.splitext(os.path.basename(filename))[0].split('_')
    for part in parts:
        if len(part) == 36 and part.count('-') == 4:  # Basic UUID format check
            return part
    return None

class Stage2Processor:
    def __init__(self, stage2_model_path, tokenizer_path, device="cuda", batch_size=4):
        self.device = device
        self.batch_size = batch_size
        self.mmtokenizer = _MMSentencePieceTokenizer(tokenizer_path)
        self.codectool = CodecManipulator("xcodec", 0, 1)
        self.codectool_stage2 = CodecManipulator("xcodec", 0, 8)
        
        print("Loading Stage 2 model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            stage2_model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        ).to(device)
        self.model.eval()

    def stage2_generate(self, prompt, batch_size=16):
        """Generate Stage 2 tokens for a given prompt"""
        codec_ids = self.codectool.unflatten(prompt, n_quantizer=1)
        codec_ids = self.codectool.offset_tok_ids(
            codec_ids,
            global_offset=self.codectool.global_offset,
            codebook_size=self.codectool.codebook_size,
            num_codebooks=self.codectool.num_codebooks,
        ).astype(np.int32)

        # Handle batched or single input
        if batch_size > 1:
            codec_list = []
            for i in range(batch_size):
                idx_begin = i * 300
                idx_end = (i + 1) * 300
                codec_list.append(codec_ids[:, idx_begin:idx_end])

            codec_ids = np.concatenate(codec_list, axis=0)
            prompt_ids = np.concatenate(
                [
                    np.tile([self.mmtokenizer.soa, self.mmtokenizer.stage_1], (batch_size, 1)),
                    codec_ids,
                    np.tile([self.mmtokenizer.stage_2], (batch_size, 1)),
                ],
                axis=1
            )
        else:
            prompt_ids = np.concatenate([
                np.array([self.mmtokenizer.soa, self.mmtokenizer.stage_1]),
                codec_ids.flatten(),
                np.array([self.mmtokenizer.stage_2])
            ]).astype(np.int32)
            prompt_ids = prompt_ids[np.newaxis, ...]

        codec_ids = torch.as_tensor(codec_ids).to(self.device)
        prompt_ids = torch.as_tensor(prompt_ids).to(self.device)
        len_prompt = prompt_ids.shape[-1]

        # Updated logits processors to match infer.py
        block_list = LogitsProcessorList([
            BlockTokenRangeProcessor(0, 46358),
            BlockTokenRangeProcessor(53526, self.mmtokenizer.vocab_size)
        ])

        # Teacher forcing generate loop with exact token generation
        for frames_idx in range(codec_ids.shape[1]):
            cb0 = codec_ids[:, frames_idx:frames_idx+1]
            prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
            input_ids = prompt_ids

            with torch.no_grad():
                stage2_output = self.model.generate(
                    input_ids=input_ids,
                    min_new_tokens=7,
                    max_new_tokens=7,
                    eos_token_id=self.mmtokenizer.eoa,
                    pad_token_id=self.mmtokenizer.eoa,
                    logits_processor=block_list,
                )

            assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, \
                f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
            prompt_ids = stage2_output

        # Process output based on batch size
        if batch_size > 1:
            output = prompt_ids.cpu().numpy()[:, len_prompt:]
            output_list = [output[i] for i in range(batch_size)]
            output = np.concatenate(output_list, axis=0)
        else:
            output = prompt_ids[0].cpu().numpy()[len_prompt:]

        return output

    def process_stage2(self, stage1_outputs, stage2_output_dir):
        """Process Stage 1 outputs through Stage 2 with simplified naming"""
        stage2_result = []
        
        for i in tqdm(range(len(stage1_outputs)), desc="Stage 2 Processing"):
            # Extract UUID and determine stem type from original filename
            uuid = extract_uuid(stage1_outputs[i])
            original_basename = os.path.basename(stage1_outputs[i])
            
            # Preserve the stem type from the input file
            if 'instrumental' in original_basename:
                stem_type = 'instrumental'
            elif 'vocal' in original_basename:
                stem_type = 'vocal'
            else:
                print(f"Cannot determine stem type from {original_basename}")
                continue
                
            # Create simple output filename
            output_filename = os.path.join(stage2_output_dir, f"{stem_type}_{uuid}.npy")
            print(f"Processing {stem_type} stem -> {output_filename}")
            
            if os.path.exists(output_filename):
                print(f'Stage 2 already completed for {output_filename}')
                stage2_result.append(output_filename)
                continue
            
            # Load and process prompt
            prompt = np.load(stage1_outputs[i]).astype(np.int32)
            output_duration = prompt.shape[-1] // 50 // 6 * 6
            num_batch = output_duration // 6
            
            try:
                # Process in appropriate batch sizes
                if num_batch <= self.batch_size:
                    output = self.stage2_generate(prompt[:, :output_duration*50], batch_size=num_batch)
                else:
                    segments = []
                    num_segments = (num_batch // self.batch_size) + (1 if num_batch % self.batch_size != 0 else 0)

                    for seg in range(num_segments):
                        start_idx = seg * self.batch_size * 300
                        end_idx = min((seg + 1) * self.batch_size * 300, output_duration*50)
                        current_batch_size = self.batch_size if seg != num_segments-1 or num_batch % self.batch_size == 0 else num_batch % self.batch_size
                        segment = self.stage2_generate(
                            prompt[:, start_idx:end_idx],
                            batch_size=current_batch_size
                        )
                        segments.append(segment)

                    output = np.concatenate(segments, axis=0)
                
                # Process ending if necessary
                if output_duration*50 != prompt.shape[-1]:
                    ending = self.stage2_generate(prompt[:, output_duration*50:], batch_size=1)
                    output = np.concatenate([output, ending], axis=0)
                
                output = self.codectool_stage2.ids2npy(output)

                # Fix invalid codes with most frequent valid codes
                fixed_output = copy.deepcopy(output)
                for i, line in enumerate(output):
                    for j, element in enumerate(line):
                        if element < 0 or element > 1023:
                            counter = Counter(line)
                            most_frequent = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                            fixed_output[i, j] = most_frequent

                np.save(output_filename, fixed_output)
                stage2_result.append(output_filename)
                
            except Exception as e:
                print(f"Error processing {stage1_outputs[i]}: {str(e)}")
                continue
            
        return stage2_result


# Modified AudioProcessor class methods
class AudioProcessor:
    @staticmethod
    def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
        """Save audio with proper scaling"""
        folder_path = os.path.dirname(path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        limit = 0.99
        max_val = wav.abs().max()
        wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
        torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

    @staticmethod
    def process_and_save_audio(stage2_results, codec_model, output_dir, device="cuda"):
        """Process Stage 2 results into audio files with simplified naming"""
        if not stage2_results or len(stage2_results) != 2:
            print(f"Expected 2 stage2 results (vocal and instrumental), got {len(stage2_results)}")
            return None
            
        recons_output_dir = os.path.join(output_dir, "recons")
        recons_mix_dir = os.path.join(recons_output_dir, 'mix')
        os.makedirs(recons_mix_dir, exist_ok=True)
        
        # Verify we have both stems and get UUID
        has_vocal = any('vocal' in r for r in stage2_results)
        has_instrumental = any('instrumental' in r for r in stage2_results)
        
        if not (has_vocal and has_instrumental):
            print(f"Missing stems: vocal={has_vocal}, instrumental={has_instrumental}")
            return None
            
        uuid = extract_uuid(stage2_results[0])  # Use same UUID for all related files
        print(f"Processing audio for UUID: {uuid}")
        
        tracks = []  # Initialize tracks list
        
        # First pass: decode and save individual stems
        for npy in stage2_results:
            try:
                codec_result = np.load(npy)
                with torch.no_grad():
                    decoded_waveform = codec_model.decode(
                        torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long)
                        .unsqueeze(0)
                        .permute(1, 0, 2)
                        .to(device)
                    )
                decoded_waveform = decoded_waveform.cpu().squeeze(0)
                
                # Simple stem naming
                stem_type = 'vocal' if 'vocal' in npy else 'instrumental'
                save_path = os.path.join(recons_output_dir, f"{stem_type}_{uuid}.mp3")
                
                AudioProcessor.save_audio(decoded_waveform, save_path, 16000)
                tracks.append(save_path)
                print(f"Saved {stem_type} stem: {save_path}")
            
            except Exception as e:
                print(f"Error processing {npy}: {str(e)}")
                continue

        # Second pass: create mix at 16kHz
        try:
            vocal_path = os.path.join(recons_output_dir, f"vocal_{uuid}.mp3")
            inst_path = os.path.join(recons_output_dir, f"instrumental_{uuid}.mp3")
            
            if not os.path.exists(vocal_path) or not os.path.exists(inst_path):
                print("Missing vocal or instrumental stem")
                return None
            
            # Create mix with simple naming
            recons_mix = os.path.join(recons_mix_dir, f"mix_{uuid}.mp3")
            
            vocal_stem, sr = sf.read(vocal_path)
            instrumental_stem, _ = sf.read(inst_path)
            
            min_length = min(len(vocal_stem), len(instrumental_stem))
            vocal_stem = vocal_stem[:min_length]
            instrumental_stem = instrumental_stem[:min_length]
            
            mix_stem = (vocal_stem + instrumental_stem) / 1
            sf.write(recons_mix, mix_stem, sr)
            print(f"Created mix: {recons_mix}")
            
        except Exception as e:
            print(f"Error mixing tracks: {e}")
        
        return recons_mix_dir

    @staticmethod
    def apply_vocoder(stage2_results, config_path, vocal_decoder_path, inst_decoder_path, 
                    output_dir, codec_model, recons_mix=None, rescale=False, cuda_idx=0):
        """Apply vocoder processing with simplified naming"""
        uuid = extract_uuid(stage2_results[0])
        
        vocal_decoder, inst_decoder = build_codec_model(config_path, vocal_decoder_path, inst_decoder_path)
        vocoder_output_dir = os.path.join(output_dir, 'vocoder')
        vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
        vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
        os.makedirs(vocoder_mix_dir, exist_ok=True)
        os.makedirs(vocoder_stems_dir, exist_ok=True)
        
        args = type('Args', (), {
            'config_path': config_path,
            'vocal_decoder_path': vocal_decoder_path,
            'inst_decoder_path': inst_decoder_path,
            'rescale': rescale,
            'cuda_idx': cuda_idx
        })()
        
        instrumental_output = None
        vocal_output = None

        # Process stems with simple naming
        for npy_file in stage2_results:
            if "instrumental" in npy_file:
                instrumental_output = process_audio(
                    npy_file,
                    os.path.join(vocoder_stems_dir, f"instrumental_{uuid}.mp3"),
                    rescale,
                    args,
                    inst_decoder,
                    codec_model
                )
            elif "vocal" in npy_file:
                vocal_output = process_audio(
                    npy_file,
                    os.path.join(vocoder_stems_dir, f"vocal_{uuid}.mp3"),
                    rescale,
                    args,
                    vocal_decoder,
                    codec_model
                )

        try:
            if instrumental_output is None or vocal_output is None:
                raise RuntimeError("Missing either instrumental or vocal output")
            
            mix_output = instrumental_output + vocal_output
            
            # Simple naming for vocoder mix
            vocoder_mix_path = os.path.join(vocoder_mix_dir, f"mix_{uuid}.mp3")
            final_output_path = os.path.join(output_dir, f"final_mix_{uuid}.mp3")
            
            AudioProcessor.save_audio(mix_output, vocoder_mix_path, 44100, rescale)
            print(f"Created upsampled mix: {vocoder_mix_path}")
            
            if not os.path.exists(recons_mix):
                raise FileNotFoundError(f"16kHz mix file not found: {recons_mix}")
            
            # Apply final processing
            replace_low_freq_with_energy_matched(
                a_file=recons_mix,
                b_file=vocoder_mix_path,
                c_file=final_output_path,
                cutoff_freq=5500.0
            )
            print(f"Created final mix: {final_output_path}")

        except Exception as e:
            print(f"Error in vocoder processing: {str(e)}")
            if instrumental_output is not None and vocal_output is not None:
                print(f"Mix shapes - Inst: {instrumental_output.shape}, Vocal: {vocal_output.shape}")
            else:
                print("One or both stems are missing")