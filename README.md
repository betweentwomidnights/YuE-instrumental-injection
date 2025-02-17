# YuE Fork - Experimental Custom Vocal Generation

⚠️ **WARNING: EXPERIMENTAL IMPLEMENTATION** ⚠️

This repository is an experimental fork of the [YuE](https://github.com/multimodal-art-projection/YuE/) project, exploring custom approaches to vocal generation using instrumental token injection. This implementation is still in early stages and likely lacks several key components for optimal performance. **Contributions and feedback are welcome!**

⚠️ **WARNING: Some outputs may be offensive** ⚠️

Sample outputs in this repo are for a fictional rapper named "yung datti" who raps in triangles and dgaf.

## Current Implementation

This fork introduces custom vocal generation scripts that attempt to inject instrumental tokens and generate vocals overtop. Our current pipeline includes audio cropping and segment-based generation, with varying levels of success.

### Command Usage

```bash
python inpainting.py \
  --input_audio_path adam_ldt.wav \
  --lyrics_file lyrics.txt \
  --genre_file genre.txt \
  --model_path m-a-p/YuE-s1-7B-anneal-en-cot \
  --stage2_model_path m-a-p/YuE-s2-1B-general \
  --codec_config ./xcodec_mini_infer/final_ckpt/config.yaml \
  --codec_checkpoint ./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth \
  --config_path ./xcodec_mini_infer/decoders/config.yaml \
  --vocal_decoder_path ./xcodec_mini_infer/decoders/decoder_131000.pth \
  --inst_decoder_path ./xcodec_mini_infer/decoders/decoder_151000.pth
```

### Key Implementation Details

- Uses step-by-step generation instead of the original `model.generate()` function
- Implements audio cropping for handling longer tracks
- Custom handling of instrumental tokens for injection
- Modified stem file naming convention

### Current Findings & Limitations

In our experiments, we've observed:

- Better coherence with 30-second segments compared to 60-second segments
  - We use `crop_audio_to_60_seconds` function but set `target_duration = 30`
  - This limitation likely stems from our incomplete context window implementation
- Mixed results with different `max_new_tokens` values
  - Experimented with 3000 and 1000 tokens
  - Inconsistent results across different attempts
- Context window implementation remains incomplete
  - Current implementation attempts to use `max_new_tokens` in multiple places
  - Need to improve sliding context window logic

### Sample Outputs

Sample outputs are provided in the root directory of this repository. Results show varying levels of:
- Lyric coherence
- Beat alignment
- Genre adherence

## Current Progress

The implementation currently achieves:
- Basic instrumental token injection
- Partial lyric generation with some timing alignment
- Vocal generation over instrumentals
- Better results with shorter segments (30s vs 60s)

However, we're still working on:
- Proper context window implementation
- Consistent vocal generation quality
- Better handling of longer audio segments
- More reliable genre adherence

## Contributing

We welcome contributions to improve this experimental implementation! Particularly interested in:
- Enhanced context window implementation
- Better handling of longer audio segments
- Documentation improvements
- Performance optimizations

## Acknowledgments

This work builds upon the original YuE project [[link to original](https://github.com/multimodal-art-projection/YuE/)]. All original credit goes to the YuE team.
