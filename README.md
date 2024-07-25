# Introduction

This is a simple example of using Meta LLaMa 3.1 to generate a simple apps

# Resouces

- [Meta LLaMa 3.1](https://llama.meta.net/)
- [Download Meta LLaMa 3.1](https://github.com/meta-llama/llama-models/)
- [convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

# Usage

1. Download the Meta LLaMa 3.1 model

```bash
% time bash resource/llama-models/models/llama3_1/download.sh 
Enter the URL from email: https://llama3-1.llamameta.net/*?Policy=XXX&Key-Pair-Id=XXX&Download-Request-ID=XXX

 **** Model list ***
 -  meta-llama-3.1-405b
 -  meta-llama-3.1-70b
 -  meta-llama-3.1-8b
 -  meta-llama-guard-3-8b
 -  prompt-guard
Choose the model to download: meta-llama-3.1-8b

 **** Available models to download: ***
 -  meta-llama-3.1-8b-instruct
 -  meta-llama-3.1-8b

Enter the list of models to download without spaces or press Enter for all: meta-llama-3.1-8b
Downloading LICENSE and Acceptable Usage Policy
...

bash resource/llama-models/models/llama3_1/download.sh  17.51s user 42.50s system 11% cpu 8:33.66 total

% du -hd1 Meta-Llama-3.1-8B 
 15G	Meta-Llama-3.1-8B

% tree Meta-Llama-3.1-8B 
Meta-Llama-3.1-8B
├── consolidated.00.pth
├── params.json
└── tokenizer.model

1 directory, 3 files
```

2. Convert the model to Hugging Face format

```bash
% python3 -m venv venv
% source venv/bin/activate
(venv) meta-llama-study % pip install transformers torch huggingface_hub tiktoken blobfile accelerate
(venv) meta-llama-study % time python3 resource/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir Meta-Llama-3.1-8B --model_size 8B --output_dir Meta-Llama-3.1-8B_hf --llama_version 3.1
Saving a LlamaTokenizerFast to llama3_1_hf.
Fetching all parameters from the checkpoint at Meta-Llama-3.1-8B.
...
Loading the checkpoint in a Llama model.
Loading checkpoint shards: 100%|██████████████████████████████████████████████████| 33/33 [00:01<00:00, 29.04it/s]
Saving in the Transformers format.
python3  --input_dir Meta-Llama-3.1-8B --model_size 8B --output_dir   3.1  48.63s user 26.62s system 132% cpu 57.008 total

(venv) meta-llama-study % du -hd1 Meta-Llama-3.1-8B_hf
 15G	Meta-Llama-3.1-8B_hf

(venv) meta-llama-study % tree Meta-Llama-3.1-8B_hf
Meta-Llama-3.1-8B_hf
├── config.json
├── generation_config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json

1 directory, 10 files
```

3. Use the converted model to generate the app

```bash
(venv) meta-llama-study % cd resource/llama.cpp
(venv) llama.cpp % time LLAMA_METAL=1 make  
...
LLAMA_METAL=1 make  79.22s user 4.51s system 96% cpu 1:26.64 total
(venv) llama.cpp % pip install -r requirements.txt
(venv) llama.cpp % time python3 convert_hf_to_gguf.py ../../Meta-Llama-3.1-8B_hf/ --outfile llama3_1-8B.gguf
...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Writing: 100%|█████████████████████████████████████████████████████████████| 16.1G/16.1G [00:36<00:00, 438Mbyte/s]
INFO:hf-to-gguf:Model successfully exported to llama3_1-8B.gguf
python3 convert_hf_to_gguf.py ../../Meta-Llama-3.1-8B_hf/ --outfile llama3_1-8B.gguf  114.26s user 101.19s system 518% cpu 41.570 total

(venv) llama.cpp % du -hd1 llama3_1-8B.gguf
 15G	llama3_1-8B.gguf
(venv) llama.cpp % ./llama-server -m ./llama3_1-8B.gguf
...
error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
...
llama_new_context_with_model: n_ctx      = 131072
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M1 Pro
ggml_metal_init: picking default device: Apple M1 Pro
ggml_metal_init: using embedded metal library
ggml_metal_init: GPU name:   Apple M1 Pro
ggml_metal_init: GPU family: MTLGPUFamilyApple7  (1007)
ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
ggml_metal_init: simdgroup reduction support   = true
ggml_metal_init: simdgroup matrix mul. support = true
ggml_metal_init: hasUnifiedMemory              = true
ggml_metal_init: recommendedMaxWorkingSetSize  = 22906.50 MB
llama_kv_cache_init:      Metal KV buffer size = 16384.00 MiB
llama_new_context_with_model: KV self size  = 16384.00 MiB, K (f16): 8192.00 MiB, V (f16): 8192.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.98 MiB
llama_new_context_with_model:      Metal compute buffer size =  8480.00 MiB
llama_new_context_with_model:        CPU compute buffer size =   264.01 MiB
llama_new_context_with_model: graph nodes  = 1030
llama_new_context_with_model: graph splits = 2
...
^Cggml_metal_free: deallocating

(venv) llama.cpp % ./llama-server -m ./llama3_1-8B.gguf -c 31072
...
.........................................................................................
llama_new_context_with_model: n_ctx      = 31072
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M1 Pro
ggml_metal_init: picking default device: Apple M1 Pro
ggml_metal_init: using embedded metal library
ggml_metal_init: GPU name:   Apple M1 Pro
ggml_metal_init: GPU family: MTLGPUFamilyApple7  (1007)
ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
ggml_metal_init: simdgroup reduction support   = true
ggml_metal_init: simdgroup matrix mul. support = true
ggml_metal_init: hasUnifiedMemory              = true
ggml_metal_init: recommendedMaxWorkingSetSize  = 22906.50 MB
llama_kv_cache_init:      Metal KV buffer size =  3884.00 MiB
llama_new_context_with_model: KV self size  = 3884.00 MiB, K (f16): 1942.00 MiB, V (f16): 1942.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.98 MiB
llama_new_context_with_model:      Metal compute buffer size =  2034.69 MiB
llama_new_context_with_model:        CPU compute buffer size =    68.69 MiB
llama_new_context_with_model: graph nodes  = 1030
llama_new_context_with_model: graph splits = 2

INFO [                    init] initializing slots | tid="0x1f3b34c00" timestamp=1721895109 n_slots=1
INFO [                    init] new slot | tid="0x1f3b34c00" timestamp=1721895109 id_slot=0 n_ctx_slot=31072
INFO [                    main] model loaded | tid="0x1f3b34c00" timestamp=1721895109
INFO [                    main] chat template | tid="0x1f3b34c00" timestamp=1721895109 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="0x1f3b34c00" timestamp=1721895109 port="8080" n_threads_http="9" hostname="127.0.0.1"
INFO [            update_slots] all slots are idle | tid="0x1f3b34c00" timestamp=1721895109
...
```
