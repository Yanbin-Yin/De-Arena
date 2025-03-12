# # import torch
# import json
# import sys

# # if torch.cuda.is_available():
# #     print(f"Total CUDA devices: {torch.cuda.device_count()}")
# #     for i in range(torch.cuda.device_count()):
# #         print(f"Device {i}: {torch.cuda.get_device_name(i)}")
# # else:
# #     print("No CUDA devices are available.")
# # existing_model_paths = {
# #     'gpt-4o-mini-2024-07-18' : "OPENAI",
# #     "gemma-2-27b-it" : "/data/shared/huggingface/hub/models--google--gemma-2-27b-it/snapshots/2d74922e8a2961565b71fd5373081e9ecbf99c08",
# #     "yi-1.5-34b-chat" : "/data/shared/huggingface/hub/models--01-ai--Yi-1.5-34B-Chat/snapshots/fa4ffba162f20948bf77c2a30eca952bf0812b7f",
# #     "llama3-8b-instruct" : "/data/shared/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/a8977699a3d0820e80129fb3c93c20fbd9972c41",
# #     "qwen1.5-14b-chat" : "/data/shared/huggingface/hub/models--Qwen--Qwen1.5-14B-Chat/snapshots/9492b22871f43e975435455f5c616c77fe7a50ec",
# #     "command-r-v01" : "/data/shared/huggingface/hub/models--CohereForAI--c4ai-command-r-v01/snapshots/16881ccde1c68bbc7041280e6a66637bc46bfe88",
# # }

# # gt_scores = {
# #     "gpt-4o-mini-2024-07-18" : 1274,
# #     "gemma-2-27b-it" : 1218,
# #     "qwen1.5-14b-chat" : 1109,
# #     "llama3-8b-instruct" : 1152,
# #     "yi-1.5-34b-chat" : 1157,
# #     "command-r-v01" : 1149,
# # }


# existing_model_paths = {
#     'gpt-4o-mini-2024-07-18' : "OPENAI",
#     'gpt4-1106' : "OPENAI",
#     'gpt3.5-turbo-0125' : "OPENAI",

#     'Athene-70b' : "/data/shared/huggingface/hub/models--Nexusflow--Athene-70B/snapshots/4b070bdb1c5fb02de52fe948da853b6980c75a41"

#     "gemma-1.1-7b-it" : "/data/shared/huggingface/hub/models--google--gemma-1.1-7b-it/snapshots/065a528791af6f57f013e8e42b7276992b45ef71",
#     "gemma-2-27b-it" : "/data/shared/huggingface/hub/models--google--gemma-2-27b-it/snapshots/2d74922e8a2961565b71fd5373081e9ecbf99c08",

#     "yi-34b-chat" : "/data/shared/huggingface/hub/models--01-ai--Yi-34B-Chat/snapshots/493781d21ad8992f4875668eff44d5af58f4e96b",
#     "yi-1.5-34b-chat" : "/data/shared/huggingface/hub/models--01-ai--Yi-1.5-34B-Chat/snapshots/fa4ffba162f20948bf77c2a30eca952bf0812b7f",

#     "mistral-7b-instruct-2" : "/data/shared/huggingface/hub/mistral-inst-7B-v0.2",
#     "mistral-8x7b-instruct-v0.1" : "/data/shared/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/1e637f2d7cb0a9d6fb1922f305cb784995190a83",

#     "llama2-13b-chat" : "/data/shared/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8",
#     "llama3-8b-instruct" : "/data/shared/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/a8977699a3d0820e80129fb3c93c20fbd9972c41",

#     "command-r-v01" : "/data/shared/huggingface/hub/models--CohereForAI--c4ai-command-r-v01/snapshots/16881ccde1c68bbc7041280e6a66637bc46bfe88",

#     "qwen1.5-14b-chat" : "/data/shared/huggingface/hub/models--Qwen--Qwen1.5-14B-Chat/snapshots/9492b22871f43e975435455f5c616c77fe7a50ec",
#     "qwen1.5-32b-chat" : "/data/shared/huggingface/hub/models--Qwen--Qwen1.5-32B-Chat/snapshots/0997b012af6ddd5465d40465a8415535b2f06cfc",
#     "qwen2-72b-instruct": "/data/shared/huggingface/hub/models--Qwen--Qwen2-72B-Instruct/models--Qwen--Qwen2-72B-Instruct/snapshots/1af63c698f59c4235668ec9c1395468cb7cd7e79",

#     "openchat-3.5" : "/data/shared/huggingface/hub/models--openchat--openchat_3.5/snapshots/c8ac81548666d3f8742b00048cbd42f48513ba62",
#     "openchat-3.5-0106" : "/data/shared/huggingface/hub/models--openchat--openchat-3.5-0106/snapshots/f3b79c43f12da94b56565c5fc5a65d40e696c876",

#     "zephyr-7b-beta": "/data/shared/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/b70e0c9a2d9e14bd1e812d3c398e5f313e93b473",

#     "vicuna-13b" : "/data/shared/huggingface/hub/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2",
#     "vicuna-33b" : "/data/shared/huggingface/hub/models--lmsys--vicuna-33b-v1.3/snapshots/ef8d6becf883fb3ce52e3706885f761819477ab4"
# }

# gt_scores = {
#     "gpt-4o-mini-2024-07-18" : 1274,
#     'gpt4-1106' : 1251,
#     "gemma-2-27b-it" : 1218,
#     "qwen2-72b-instruct": 1187,
#     "yi-1.5-34b-chat" : 1157,
#     "llama3-8b-instruct" : 1152,
#     "command-r-v01" : 1149,
#     "qwen1.5-32b-chat" : 1125,
#     "mistral-8x7b-instruct-v0.1" : 1114,
#     "yi-34b-chat" : 1111,
#     "qwen1.5-14b-chat" : 1109,
#     'gpt3.5-turbo-0125' : 1106,
#     "openchat-3.5-0106" : 1092,
#     "vicuna-33b" : 1091,
#     "gemma-1.1-7b-it" : 1084,
#     "openchat-3.5" : 1076,
#     "mistral-7b-instruct-2" : 1072,
#     "llama2-13b-chat" : 1063,
#     "zephyr-7b-beta" : 1053,
#     "vicuna-13b" : 1042,
# }

import torch
import json
import sys

if torch.cuda.is_available():
    print(f"Total CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices are available.")

existing_model_paths = {
    'gpt-4o-mini-2024-07-18' : "OPENAI", # 1
    'gpt4-1106' : "OPENAI",
    'gpt3.5-turbo-0125' : "OPENAI", # 2
    "o1-preview" : "OPENAI",
    "o1-mini" : "OPENAI",
    "ChatGPT-4o-latest (2024-09-03)" : "OPENAI",
    "gpt-4o-2024-08-06" : "OPENAI", # 3
    "gpt-4-turbo-2024-04-09" : "OPENAI",
    "gpt-4o-2024-05-13" : "OPENAI",

    "claude-3.5-sonnet" : "Claude",
    "claude-3-opus" : "Claude",
    "claude-3-sonnet" : "Claude",
    "claude-3-haiku" : "Claude", # 1
    "claude-2.0" : "Claude",
    "claude-2.1" : "Claude",
    "claude-3.5-sonnet-20241022" : "Claude",

    "gemini-1.5-flash-001" : "gemini",
    "gemini-1.5-pro-001" : "gemini",
    "gemini-1.0-pro-001" : "gemini",

    "yi-lightning": "yi-lightning",
    "glm-4-plus": "glm-4-plus",

    'athene-70b' : "/mbz/shared/huggingface/hub/models--Nexusflow--Athene-70B/snapshots/4b070bdb1c5fb02de52fe948da853b6980c75a41",
    # "stripedhyena-nous-7b" : "/mbz/shared/huggingface/hub/models--togethercomputer--StripedHyena-Nous-7B/snapshots/d8a5c9f5698bf7253c8e76b41efc3d2e65abfd09",
    "gemma-2-9b-it-simpo" : "/mbz/shared/huggingface/hub/models--princeton-nlp--gemma-2-9b-it-SimPO/snapshots/8c87091f412e3aa6f74f66bd86c57fb81cbc3fde",
    # "yi-34b-chat" : "/mbz/shared/huggingface/hub/models--01-ai--Yi-34B-Chat/snapshots/493781d21ad8992f4875668eff44d5af58f4e96b",
    "yi-1.5-34b-chat" : "/mbz/shared/huggingface/hub/models--01-ai--Yi-1.5-34B-Chat/snapshots/fa4ffba162f20948bf77c2a30eca952bf0812b7f",
    "llama3-8b-instruct" : "/mbz/shared/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa",
    "qwen1.5-14b-chat" : "/mbz/shared/huggingface/hub/models--Qwen--Qwen1.5-14B-Chat/snapshots/9492b22871f43e975435455f5c616c77fe7a50ec",
    "qwen1.5-32b-chat" : "/mbz/shared/huggingface/hub/models--Qwen--Qwen1.5-32B-Chat/snapshots/0997b012af6ddd5465d40465a8415535b2f06cfc",
    "qwen2-72b-instruct": "/mbz/shared/huggingface/hub/models--Qwen--Qwen2-72B-Instruct/snapshots/fddbbd7b69a1fd7cf9b659203b37ae3eb89059e1",
    "qwen1.5-72b-chat" : "/mbz/shared/huggingface/hub/models--Qwen--Qwen1.5-72B-Chat/snapshots/d341a6f2cb937e7a830ecbe3ab7b87215bc3a6b0",
    "openchat-3.5" : "/mbz/shared/huggingface/hub/models--openchat--openchat_3.5/snapshots/0fc98e324280bc4bf5d2c30ecf7b97b84fb8a19b",
    "openchat-3.5-0106" : "/mbz/shared/huggingface/hub/models--openchat--openchat-3.5-0106/snapshots/ff058fda49726ecf4ea53dc1635f917cdb8ba36b",
    "vicuna-13b" : "/mbz/shared/huggingface/hub/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2",
    "llama-3-70b-instruct": "/mbz/shared/huggingface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/1480bb72e06591eb87b0ebe2c8853127f9697bae",
    "openassistant-pythia-12b": "/mbz/shared/huggingface/hub/models--OpenAssistant--oasst-sft-1-pythia-12b/snapshots/293df535fe7711a5726987fc2f17dfc87de452a1",
    "starling-lm-7b-beta": "/mbz/shared/huggingface/hub/models--Nexusflow--Starling-LM-7B-beta/snapshots/39a4d501472dfede947ca5f4c5af0c1896d0361b",

    "zephyr-7b-beta": "/mbz/shared/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/20e1a5880bb00a7571542fe3fe6cb2dcb4816eee",
    "vicuna-33b": "/mbz/shared/huggingface/hub/models--lmsys--vicuna-33b-v1.3/snapshots/ef8d6becf883fb3ce52e3706885f761819477ab4",
    "google-gemma-2-9b-it": "/mbz/shared/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "vicuna-7b": "/mbz/shared/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d",
    "meta-llama-3.1-70b-instruct": "/mbz/shared/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693",
    "starling-lm-7b-alpha" : "/mbz/shared/huggingface/hub/models--berkeley-nest--Starling-LM-7B-alpha/snapshots/1dddf3b95bc1391f6307299eb1c162c194bde9bd",
    "koala-13b" : "/mbz/shared/huggingface/hub/models--TheBloke--koala-13B-HF/snapshots/b20f96a0171ce4c0fa27d6048215ebe710521587",

    "gemma-1.1-2b-it" : "/mbz/shared/huggingface/hub/models--google--gemma-1.1-2b-it/snapshots/d750f5eceb83e978c09e2b3597c2a8784e381022",
    "gemma-1.1-7b-it" : "/mbz/shared/huggingface/hub/models--google--gemma-1.1-7b-it/snapshots/065a528791af6f57f013e8e42b7276992b45ef71",
    "gemma-2-27b-it" : "/mbz/shared/huggingface/hub/models--google--gemma-2-27b-it/snapshots/aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0",
    "gemma-2-2b-it" : "/mbz/shared/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
    "gemma-2b-it" : "/mbz/shared/huggingface/hub/models--google--gemma-2b-it/snapshots/4cf79afa15bef73c0b98ff5937d8e57d6071ef71",
    "gemma-7b-it" : "/mbz/shared/huggingface/hub/models--google--gemma-7b-it/snapshots/9c5798d27f588501ce1e108079d2a19e4c3a2353",

    "mistral-7b-instruct-2" : "/mbz/shared/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c",
    "mistral-7b-instruct-1" : "/mbz/shared/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe",
    "mistral-8x7b-instruct-v0.1" : "/mbz/shared/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1",

    "llama2-13b-chat" : "/mbz/shared/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8",
    "llama2-7b-chat" : "/mbz/shared/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",

    "command-r-(04-2024)" : "/mbz/shared/huggingface/hub/models--CohereForAI--c4ai-command-r-v01/snapshots/8089a087fb3186647a1be567c35184c32e4a3cd2",
    "command-r-(08-2024)" : "/mbz/shared/huggingface/hub/models--CohereForAI--c4ai-command-r-08-2024/snapshots/280b5c1632407e0b90375eb1117a46b388da35c3",

    "qwen1.5-4b-chat" : "/mbz/shared/huggingface/hub/models--Qwen--Qwen1.5-4B-Chat/snapshots/a7a4d4945d28bac955554c9abd2f74a71ebbf22f",

    "meta-llama-3.1-8b-instruct" : "/mbz/shared/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",

    "jamba-1.5-mini" : "/mbz/shared/huggingface/hub/models--ai21labs--AI21-Jamba-1.5-Mini/snapshots/1840d3373c51e4937f4dbaaaaf8cac1427b46858",
    "llama-3.2-3b-it": "/mbz/shared/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72",
    "nemotron-70b": "/mbz/shared/huggingface/hub/models--nvidia--Llama-3.1-Nemotron-70B-Instruct-HF/snapshots/fac73d3507320ec1258620423469b4b38f88df6e",
    "ministral-8b-it": "/mbz/shared/huggingface/hub/models--mistralai--Ministral-8B-Instruct-2410/snapshots/05894f4c1a269ccce053c854da977fe06cad2d17", # 42
    "smollm2-1.7b": "/mbz/shared/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/84e8f3e31df252d9cdd9a1da5aa0adefe4dac0ea",
    "qwen2.5-1.5b": "/mbz/shared/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "llama-3.2-1b-it": "/mbz/shared/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/e9f8effbab1cbdc515c11ee6e098e3d5a9f51e14"
    #"ministral-8b-it": "/mbz/shared/huggingface/hub/models--prince-canuma--Ministral-8B-Instruct-2410-HF/snapshots/e0a14d7a6a8a1d1e5bef1a77a42e86e8bcae0ee7",
}

existing_model_paths_QS = {
    'athene-70b' : "/mbz/shared/huggingface/hub/models--Nexusflow--Athene-70B/snapshots/4b070bdb1c5fb02de52fe948da853b6980c75a41",
    "gemma-2-27b-it" : "/mbz/shared/huggingface/hub/models--google--gemma-2-27b-it/snapshots/aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0",
    "gemma-2-9b-it-simpo" : "/mbz/shared/huggingface/hub/models--princeton-nlp--gemma-2-9b-it-SimPO/snapshots/8c87091f412e3aa6f74f66bd86c57fb81cbc3fde",
    "google-gemma-2-9b-it": "/mbz/shared/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "qwen2-72b-instruct": "/mbz/shared/huggingface/hub/models--Qwen--Qwen2-72B-Instruct/snapshots/fddbbd7b69a1fd7cf9b659203b37ae3eb89059e1",
    "command-r-(08-2024)" : "/mbz/shared/huggingface/hub/models--CohereForAI--c4ai-command-r-08-2024/snapshots/280b5c1632407e0b90375eb1117a46b388da35c3",
    "yi-1.5-34b-chat" : "/mbz/shared/huggingface/hub/models--01-ai--Yi-1.5-34B-Chat/snapshots/fa4ffba162f20948bf77c2a30eca952bf0812b7f",
    "mistral-8x7b-instruct-v0.1": "/mbz/shared/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1",
    "qwen1.5-14b-chat": "/mbz/shared/huggingface/hub/models--Qwen--Qwen1.5-14B-Chat/snapshots/9492b22871f43e975435455f5c616c77fe7a50ec",
    "openchat-3.5-0106": "/mbz/shared/huggingface/hub/models--openchat--openchat-3.5-0106/snapshots/ff058fda49726ecf4ea53dc1635f917cdb8ba36b",
    "starling-lm-7b-alpha": "/mbz/shared/huggingface/hub/models--berkeley-nest--Starling-LM-7B-alpha/snapshots/1dddf3b95bc1391f6307299eb1c162c194bde9bd",
    "mistral-7b-instruct-2": "/mbz/shared/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/3ad372fc79158a2148299e3318516c786aeded6c",
    "vicuna-13b": "/mbz/shared/huggingface/hub/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2",
    "mistral-7b-instruct-1": "/mbz/shared/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe",
    "koala-13b": "/mbz/shared/huggingface/hub/models--TheBloke--koala-13B-HF/snapshots/b20f96a0171ce4c0fa27d6048215ebe710521587",
}

gt_scores = {
    "o1-preview" : 1355,
    "o1-mini" : 1324,
    "claude-3.5-sonnet" : 1269,
    "claude-3-opus" : 1248,
    "claude-3-sonnet" : 1201,
    "claude-3-haiku" : 1179,
    "claude-2.0" : 1132,
    "claude-2.1" : 1118,
    "ChatGPT-4o-latest (2024-09-03)" : 1335,
    "gpt-4o-mini-2024-07-18" : 1273,
    "gpt-4o-2024-08-06" : 1263,
    "gpt-4-Turbo-2024-04-09" : 1257,
    "gpt-4o-2024-05-13" : 1285,
    'gpt4-1106' : 1251,
    'gpt3.5-turbo-0125' : 1106,
    "athene-70b": 1250,
    "jamba-1.5-mini": 1176,
    "gemma-2b-it" : 989,
    "gemma-7b-it" : 1038,
    "gemma-1.1-7b-it" : 1084,
    "gemma-2-27b-it" : 1217,
    "google-gemma-2-9b-it" : 1188,
    "gemma-2-2b-it": 1135,
    "gemma-1.1-2b-it" : 1021,
    "yi-34b-chat" : 1111,
    "yi-1.5-34b-chat" : 1157,
    "mistral-7b-instruct-1" : 1008,
    "mistral-7b-instruct-2" : 1114,
    "mistral-8x7b-instruct-v0.1" : 1114,
    "meta-llama-3.1-8b-instruct" : 1161,
    "llama2-13b-chat" : 1063,
    "llama3-8b-instruct" : 1152,
    "llama-2-7b-chat" : 1037,
    "meta-llama-3.1-70b-instruct" : 1248,
    "meta-llama-3-70b-instruct" : 1206,
    "command-r-(04-2024)" : 1149,
    "command-r-(08-2024)" : 1171,
    "qwen1.5-4B-chat" : 988,
    "qwen1.5-14b-chat" : 1095,
    "qwen1.5-32b-chat" : 1110,
    "qwen1.5-72b-chat" : 1147,
    "qwen2-72b-instruct": 1170,
    "openchat-3.5" : 1076,
    "openchat-3.5-0106" : 1092,
    "vicuna-13b" : 1042,
    "vicuna-33b" : 1091,
    "vicuna-7b" : 1005,
    "zephyr-7b-alpha" : 1041,
    "zephyr-7b-beta" : 1053,
    "tulu-2-dpo-70b" : 1099,
    "starling-lm-7b-alpha" : 1088,
    "starling-lm-7b-beta" : 1119,
    "codellama-34b-instruct" : 1043,
    "gemma-2-9b-it-simpo" : 1216,
    "openassistant-pythia-12b" : 894,
}
