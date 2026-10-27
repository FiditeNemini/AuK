<!-- omit in toc -->
# Shared Model Cards
<!-- omit in toc -->
### **Prerequisites of using**
- This document is serving as a quick lookup table for the community fine-tuning results, with various language support.
- The models in this repository are open source and are based on voluntary contributions from contributors.
- The use of models must be conditioned on respect for the respective creators. The convenience brought comes from their efforts.
<!-- omit in toc -->
### **Welcome to share here**
- Have a fine-tuned result: model checkpoint (pruned best to facilitate inference, i.e. leave only `ema_model_state_dict`) and the corresponding config file.
- Host a public model repository ([huggingface](https://huggingface.co/new) or [ModelScope](https://www.modelscope.cn)) and upload the model related files.
- Make a pull request adding a model card to the current page, i.e. `docs/SHARED.md`.
<!-- omit in toc -->
### Supported Languages
- [French](#french)
    - [AuK Base @ fr @ Iceky](#auk-base--fr--iceky)
## French
#### AuK Base @ fr @ Iceky
|Model|🤗Hugging Face|Data|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|AuK Base + LoRA|[ckpt & config](https://huggingface.co/Iceky/AuK-French-LoRA)|[VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) & [MLS](https://www.openslr.org/94/), 327k utterances|mit|
```bash
Model: hf://Iceky/AuK-French-LoRA/model_last.safetensors
Config: hf://Iceky/AuK-French-LoRA/config.yaml
```
- Adapter: LoRA rank 32, merged into the backbone.
- Evaluation on the French subset of X-Voice-Testset (500 prompts): WER 9.028, SIM 0.7464, UTMOS 3.4752. WER with `whisper-large-v3` and text normalisation, SIM with WavLM-ECAPA, UTMOS with SpeechMOS; generation at 32 NFE and CFG 2.0 with a fixed seed. 
- Fine-tuned by [@Iceky](https://huggingface.co/Iceky)
- Any improvements are welcome.
