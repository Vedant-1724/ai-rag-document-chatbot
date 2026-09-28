# Retrieval-Augmented Generation Documentation

> This document was collected from official documentation sources for use as a Retrieval-Augmented Generation (RAG) corpus.

## Sources


- [Hugging Face RAG](https://huggingface.co/docs/transformers/main/model_doc/rag)
- [Hugging Face RAG with Chat Templates, Tools and Documents](https://huggingface.co/docs/transformers/main/chat_template_tools_and_documents)
- [Hugging Face Advanced Chat Templates](https://huggingface.co/docs/transformers/main/chat_template_advanced)
- [Hugging Face Tokenizer Documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer)


---


## Hugging Face RAG

**Source:** https://huggingface.co/docs/transformers/main/model_doc/rag

Transformers documentation
RAG
Transformers
🏡 View all docs
AWS Trainium & Inferentia
Accelerate
Argilla
AutoTrain
Bitsandbytes
CLI
Chat UI
Dataset viewer
Datasets
Deploying on AWS
Diffusers
Distilabel
Evaluate
Google Cloud
Google TPUs
Gradio
Hub
Hub Python Library
Huggingface.js
Inference Endpoints (dedicated)
Inference Providers
Kernels
LeRobot
Leaderboards
Lighteval
Microsoft Azure
OpenEnv
Optimum
PEFT
Reachy Mini
Safetensors
Sentence Transformers
TRL
Tasks
Text Embeddings Inference
Text Generation Inference
Tokenizers
Trackio
Transformers
Transformers.js
Xet
smolagents
timm
Search documentation
main
v5.17.0
v5.15.1
v5.14.0
v5.13.1
v5.12.0
v5.11.0
v5.10.4
v5.9.0
v5.8.1
v5.7.0
v5.6.2
v5.5.4
v5.4.0
v5.3.0
v5.2.0
v5.1.0
v5.0.0
v4.57.6
v4.56.2
v4.55.4
v4.53.3
v4.52.3
v4.51.3
v4.50.0
v4.49.0
v4.48.2
v4.47.1
v4.46.3
v4.45.2
v4.44.2
v4.43.4
v4.42.4
v4.41.2
v4.40.2
v4.39.3
v4.38.2
v4.37.2
v4.36.1
v4.35.2
v4.34.1
v4.33.3
v4.32.1
v4.31.0
v4.30.0
v4.29.1
v4.28.1
v4.27.2
v4.26.1
v4.25.1
v4.24.0
v4.23.1
v4.22.2
v4.21.3
v4.20.1
v4.19.4
v4.18.0
v4.17.0
v4.16.2
v4.15.0
v4.14.1
v4.13.0
v4.12.5
v4.11.3
v4.10.1
v4.9.2
v4.8.2
v4.7.0
v4.6.0
v4.5.1
v4.4.2
v4.3.3
v4.2.2
v4.1.1
v4.0.1
v3.5.1
v3.4.0
v3.3.1
v3.2.0
v3.1.0
v3.0.2
v2.11.0
v2.10.0
v2.9.1
v2.8.0
v2.7.0
v2.6.0
v2.5.1
v2.4.1
v2.3.0
v2.2.2
v2.1.1
v2.0.0
v1.2.0
v1.1.0
v1.0.0
doc-builder-html
AR
DE
EN
ES
FR
HI
IT
JA
KO
PT
RO
TE
TR
ZH
You are viewing
main
version, which requires
installation from source
. If you'd like
 regular pip install, checkout the latest stable version (
v5.17.0
).
Join the Hugging Face community
and get access to the augmented documentation experience
Collaborate on models, datasets and Spaces
Faster examples with accelerated inference
Switch between documentation themes
Sign Up
to get started
This model was published in HF papers on 2020-05-22 and contributed to Hugging Face Transformers on 2020-11-16.
Copy page
RAG
Retrieval-Augmented Generation (RAG)
combines a pretrained language model (parametric memory) with access to an external data source (non-parametric memory) by means of a pretrained neural retriever. RAG fetches relevant passages and conditions its generation on them during inference. This often makes the answers more factual and lets you update knowledge by changing the index instead of retraining the whole model.
You can find all the original RAG checkpoints under the
AI at Meta
organization.
This model was contributed by
ola13
.
Click on the RAG models in the right sidebar for more examples of how to apply RAG to different language tasks.
The examples below demonstrate how to generate text with
AutoModel
.
AutoModel
Copied
from
transformers
import
RagRetriever, RagSequenceForGeneration, RagTokenizer

tokenizer = RagTokenizer.from_pretrained(
"facebook/rag-sequence-nq"
)
retriever = RagRetriever.from_pretrained(
"facebook/rag-sequence-nq"
, dataset=
"wiki_dpr"
, index_name=
"compressed"
)

model = RagSequenceForGeneration.from_pretrained(
"facebook/rag-sequence-nq"
,
 retriever=retriever,
 attn_implementation=
"flash_attention_2"
,
 device_map=
"auto"
,
)
inputs = tokenizer(
"How many people live in Paris?"
, return_tensors=
"pt"
).to(model.device)
generated = model.generate(input_ids=inputs[
"input_ids"
])
print
(tokenizer.batch_decode(generated, skip_special_tokens=
True
)[
0
])
Quantization reduces memory by storing weights in lower precision. See the
Quantization
overview for supported backends.
The example below uses
bitsandbytes
to quantize the weights to 4-bits.
Copied
import
torch
from
transformers
import
BitsAndBytesConfig, RagRetriever, RagSequenceForGeneration, RagTokenizer

bnb = BitsAndBytesConfig(load_in_4bit=
True
, bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = RagTokenizer.from_pretrained(
"facebook/rag-sequence-nq"
)
retriever = RagRetriever.from_pretrained(
"facebook/rag-sequence-nq"
, dataset=
"wiki_dpr"
, index_name=
"compressed"
)

model = RagSequenceForGeneration.from_pretrained(
"facebook/rag-sequence-nq"
,
 retriever=retriever,
 quantization_config=bnb,
 device_map=
"auto"
,
)
inputs = tokenizer(
"How many people live in Paris?"
, return_tensors=
"pt"
).to(model.device)
generated = model.generate(input_ids=inputs[
"input_ids"
])
print
(tokenizer.batch_decode(generated, skip_special_tokens=
True
)[
0
])
RagConfig
class
transformers.
RagConfig
<
source
>
(
transformers_version
: str | None = None
architectures
: list[str] | None = None
output_hidden_states
: bool | None = False
return_dict
: bool | None = True
dtype
: str | torch.dtype | None = None
chunk_size_feed_forward
: int = 0
id2label
: dict[int, str] | dict[str, str] | None = None
label2id
: dict[str, int] | dict[str, str] | None = None
problem_type
: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None
is_encoder_decoder
: bool = True
vocab_size
: int | None = None
prefix
: str | None = None
bos_token_id
: int | None = None
pad_token_id
: int | None = None
eos_token_id
: int | list[int] | None = None
decoder_start_token_id
: int | None = None
title_sep
: str = ' / '
doc_sep
: str = ' // '
n_docs
: int = 5
max_combined_length
: int = 300
retrieval_vector_size
: int = 768
retrieval_batch_size
: int = 8
dataset
: str = 'wiki_dpr'
dataset_split
: str = 'train'
index_name
: str = 'compressed'
index_path
: str | None = None
passages_path
: str | None = None
use_dummy_dataset
: bool = False
reduce_loss
: bool = False
label_smoothing
: float = 0.0
do_deduplication
: bool = True
exclude_bos_score
: bool = False
do_marginalize
: bool = False
output_retrieved
: bool = False
use_cache
: bool = True
dataset_revision
: str | None = None
)
Parameters
is_encoder_decoder
(
bool
,
optional
, defaults to
True
) —
Whether the model is used as an encoder/decoder or not.
vocab_size
(
int
,
optional
) —
Vocabulary size of the model. Defines the number of different tokens that can be represented by the
input_ids
.
prefix
(
str
,
optional
) —
A string prefix prepended to every input before passing to the generator model.
bos_token_id
(
int
,
optional
) —
Token id used for beginning-of-stream in the vocabulary.
pad_token_id
(
int
,
optional
) —
Token id used for padding in the vocabulary.
eos_token_id
(
Union[int, list[int]]
,
optional
) —
Token id used for end-of-stream in the vocabulary.
decoder_start_token_id
(
int
,
optional
) —
If an encoder-decoder model starts decoding with a different token than
bos
, the id of that token.
title_sep
(
str
,
optional
, defaults to
" / "
) —
Separator inserted between the title and the text of the retrieved document when calling
RagRetriever
.
doc_sep
(
str
,
optional
, defaults to
" // "
) —
Separator inserted between the text of the retrieved document and the original input when calling
RagRetriever
.
n_docs
(
int
,
optional
, defaults to 5) —
Number of documents to retrieve.
max_combined_length
(
int
,
optional
, defaults to 300) —
Max length of contextualized input returned by
__call__()
.
retrieval_vector_size
(
int
,
optional
, defaults to 768) —
Dimensionality of the document embeddings indexed by
RagRetriever
.
retrieval_batch_size
(
int
,
optional
, defaults to 8) —
Retrieval batch size, defined as the number of queries issues concurrently to the faiss index encapsulated
RagRetriever
.
dataset
(
str
,
optional
, defaults to
"wiki_dpr"
) —
A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and ids
using
datasets.list_datasets()
).
dataset_split
(
str
,
optional
, defaults to
"train"
) —
Which split of the
dataset
to load.
index_name
(
str
,
optional
, defaults to
"compressed"
) —
The index name of the index associated with the
dataset
. One can choose between
"legacy"
,
"exact"
and
"compressed"
.
index_path
(
str
,
optional
) —
The path to the serialized faiss index on disk.
passages_path
(
str
,
optional
) —
A path to text passages compatible with the faiss index. Required if using
LegacyIndex
use_dummy_dataset
(
bool
,
optional
, defaults to
False
) —
Whether to load a “dummy” variant of the dataset specified by
dataset
.
reduce_loss
(
bool
,
optional
, defaults to
False
) —
Whether or not to reduce the NLL loss using the
torch.Tensor.sum
operation.
label_smoothing
(
float
,
optional
, defaults to 0.0) —
Only relevant if
return_loss
is set to
True
. Controls the
epsilon
parameter value for label smoothing
in the loss calculation. If set to 0, no label smoothing is performed.
do_deduplication
(
bool
,
optional
, defaults to
True
) —
Whether or not to deduplicate the generations from different context documents for a given input. Has to be
set to
False
if used while training with distributed backend.
exclude_bos_score
(
bool
,
optional
, defaults to
False
) —
Whether or not to disregard the BOS token when computing the loss.
do_marginalize
(
bool
,
optional
, defaults to
False
) —
If
True
, the logits are marginalized over all documents by making use of
torch.nn.functional.log_softmax
.
output_retrieved
(
bool
,
optional
, defaults to
False
) —
If set to
True
,
retrieved_doc_embeds
,
retrieved_doc_ids
,
context_input_ids
and
context_attention_mask
are returned. See returned tensors for more detail.
use_cache
(
bool
,
optional
, defaults to
True
) —
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if
config.is_decoder=True
or when the model is a decoder-only generative model.
dataset_revision
(
str
,
optional
,) —
The revision (commit hash, tag, or branch) of the Hugging Face dataset used for retrieval.
This is the configuration class to store the configuration of a RagModel. It is used to instantiate a Rag
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the
Configuration objects inherit from
PreTrainedConfig
and can be used to control the model outputs. Read the
documentation from
PreTrainedConfig
for more information.
from_question_encoder_generator_configs
<
source
>
(
question_encoder_config
: PreTrainedConfig
generator_config
: PreTrainedConfig
**kwargs
)
→
EncoderDecoderConfig
Returns
EncoderDecoderConfig
An instance of a configuration object
Instantiate a
EncoderDecoderConfig
(or a derived class) from a pre-trained encoder model configuration and
decoder model configuration.
RagTokenizer
class
transformers.
RagTokenizer
<
source
>
(
question_encoder
generator
)
Rag specific outputs
class
transformers.models.rag.modeling_rag.
RetrievAugLMMarginOutput
<
source
>
(
loss
: typing.Optional[torch.FloatTensor] = None
logits
: typing.Optional[torch.FloatTensor] = None
doc_scores
: typing.Optional[torch.FloatTensor] = None
past_key_values
: transformers.cache_utils.Cache | None = None
retrieved_doc_embeds
: typing.Optional[torch.FloatTensor] = None
retrieved_doc_ids
: typing.Optional[torch.LongTensor] = None
context_input_ids
: typing.Optional[torch.LongTensor] = None
context_attention_mask
: typing.Optional[torch.LongTensor] = None
question_encoder_last_hidden_state
: typing.Optional[torch.FloatTensor] = None
question_enc_hidden_states
: tuple[torch.FloatTensor, ...] | None = None
question_enc_attentions
: tuple[torch.FloatTensor, ...] | None = None
generator_enc_last_hidden_state
: typing.Optional[torch.FloatTensor] = None
generator_enc_hidden_states
: tuple[torch.FloatTensor, ...] | None = None
generator_enc_attentions
: tuple[torch.FloatTensor, ...] | None = None
generator_dec_hidden_states
: tuple[torch.FloatTensor, ...] | None = None
generator_dec_attentions
: tuple[torch.FloatTensor, ...] | None = None
generator_cross_attentions
: tuple[torch.FloatTensor, ...] | None = None
)
Parameters
loss
(
torch.FloatTensor
of shape
(1,)
,
optional
, returned when
labels
is provided) —
Language modeling loss.
logits
(
torch.FloatTensor
of shape
(batch_size, sequence_length, config.vocab_size)
) —
Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
each vocabulary token.
doc_scores
(
torch.FloatTensor
of shape
(batch_size, config.n_docs)
) —
Score between each retrieved document embeddings (see
retrieved_doc_embeds
) and
question_encoder_last_hidden_state
.
past_key_values
(
Cache
,
optional
, returned when
use_cache=True
is passed or when
config.use_cache=True
) —
It is a
Cache
instance. For more details, see our
kv cache guide
.
Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
(see
past_key_values
input) to speed up sequential decoding.
retrieved_doc_embeds
(
torch.FloatTensor
of shape
(batch_size, config.n_docs, hidden_size)
,
optional
, returned when
output_retrieved=True
) —
Embedded documents retrieved by the retriever. Is used with
question_encoder_last_hidden_state
to compute
the
doc_scores
.
retrieved_doc_ids
(
torch.LongTensor
of shape
(batch_size, config.n_docs)
,
optional
, returned when
output_retrieved=True
) —
The indexes of the embedded documents retrieved by the retriever.
context_input_ids
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
context_attention_mask
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Attention mask post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever.
question_encoder_last_hidden_state
(
torch.FloatTensor
of shape
(batch_size, sequence_length, hidden_size)
,
optional
) —
Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
model.
question_enc_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) —
Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
question_enc_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) —
Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_enc_last_hidden_state
(
torch.FloatTensor
of shape
(batch_size, sequence_length, hidden_size)
,
optional
) —
Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
generator_enc_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) —
Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
generator_enc_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) —
Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_dec_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) —
Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
generator_dec_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) —
Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_cross_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) —
Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
weighted average in the cross-attention heads.
Base class for retriever augmented marginalized models outputs.
class
transformers.models.rag.modeling_rag.
RetrievAugLMOutput
<
source
>
(
logits
: typing.Optional[torch.FloatTensor] = None
doc_scores
: typing.Optional[torch.FloatTensor] = None
past_key_values
: transformers.cache_utils.Cache | None = None
retrieved_doc_embeds
: typing.Optional[torch.FloatTensor] = None
retrieved_doc_ids
: typing.Optional[torch.LongTensor] = None
context_input_ids
: typing.Optional[torch.LongTensor] = None
context_attention_mask
: typing.Optional[torch.LongTensor] = None
question_encoder_last_hidden_state
: typing.Optional[torch.FloatTensor] = None
question_enc_hidden_states
: tuple[torch.FloatTensor, ...] | None = None
question_enc_attentions
: tuple[torch.FloatTensor, ...] | None = None
generator_enc_last_hidden_state
: typing.Optional[torch.FloatTensor] = None
generator_enc_hidden_states
: tuple[torch.FloatTensor, ...] | None = None
generator_enc_attentions
: tuple[torch.FloatTensor, ...] | None = None
generator_dec_hidden_states
: tuple[torch.FloatTensor, ...] | None = None
generator_dec_attentions
: tuple[torch.FloatTensor, ...] | None = None
generator_cross_attentions
: tuple[torch.FloatTensor, ...] | None = None
)
Parameters
logits
(
torch.FloatTensor
of shape
(batch_size, sequence_length, config.vocab_size)
) —
Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
each vocabulary token.
doc_scores
(
torch.FloatTensor
of shape
(batch_size, config.n_docs)
) —
Score between each retrieved document embeddings (see
retrieved_doc_embeds
) and
question_encoder_last_hidden_state
.
past_key_values
(
Cache
,
optional
, returned when
use_cache=True
is passed or when
config.use_cache=True
) —
It is a
Cache
instance. For more details, see our
kv cache guide
.
Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
(see
past_key_values
input) to speed up sequential decoding.
retrieved_doc_embeds
(
torch.FloatTensor
of shape
(batch_size, config.n_docs, hidden_size)
,
optional
, returned when
output_retrieved=True
) —
Embedded documents retrieved by the retriever. Is used with
question_encoder_last_hidden_state
to compute
the
doc_scores
.
retrieved_doc_ids
(
torch.LongTensor
of shape
(batch_size, config.n_docs)
,
optional
, returned when
output_retrieved=True
) —
The indexes of the embedded documents retrieved by the retriever.
context_input_ids
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
context_attention_mask
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Attention mask post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever.
question_encoder_last_hidden_state
(
torch.FloatTensor
of shape
(batch_size, sequence_length, hidden_size)
,
optional
) —
Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
model.
question_enc_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) —
Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
question_enc_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) —
Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_enc_last_hidden_state
(
torch.FloatTensor
of shape
(batch_size, sequence_length, hidden_size)
,
optional
) —
Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
generator_enc_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) —
Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
generator_enc_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) —
Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_dec_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) —
Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
generator_dec_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) —
Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_cross_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) —
Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
weighted average in the cross-attention heads.
RagRetriever
class
transformers.
RagRetriever
<
source
>
(
config
question_encoder_tokenizer
generator_tokenizer
index
= None
init_retrieval
= True
)
Parameters
config
(
RagConfig
) —
The configuration of the RAG model this Retriever is used with. Contains parameters indicating which
Index
to build. You can load your own custom dataset with
config.index_name="custom"
or use a canonical
one (default) from the datasets library with
config.index_name="wiki_dpr"
for example.
question_encoder_tokenizer
(
PreTrainedTokenizer
) —
The tokenizer that was used to tokenize the question. It is used to decode the question and then use the
generator_tokenizer.
generator_tokenizer
(
PreTrainedTokenizer
) —
The tokenizer used for the generator part of the RagModel.
index
(
Index
, optional, defaults to the one defined by the configuration) —
If specified, use this index instead of the one built using the configuration
Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents
contents, and it formats them to be used with a RagModel.
Examples:
Copied
>>>
# To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
>>>
from
transformers
import
RagRetriever
>>>
retriever = RagRetriever.from_pretrained(
...
"facebook/rag-sequence-nq"
, dataset=
"wiki_dpr"
, index_name=
"compressed"
...
)
>>>
# To load your own indexed dataset built with the datasets library.
>>>
from
transformers
import
RagRetriever
>>>
dataset = (
...
...
...
)
# dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a supported index (e.g., Faiss or other index types depending on your setup)
>>>
retriever = RagRetriever.from_pretrained(
"facebook/rag-sequence-nq"
, indexed_dataset=dataset)
>>>
# To load your own indexed dataset built with the datasets library that was saved on disk.
>>>
from
transformers
import
RagRetriever
>>>
dataset_path =
"path/to/my/dataset"
# dataset saved via *dataset.save_to_disk(...)*
>>>
index_path =
"path/to/my/index"
# index saved via *dataset.get_index("embeddings").save(...)*
>>>
retriever = RagRetriever.from_pretrained(
...
"facebook/rag-sequence-nq"
,
...
index_name=
"custom"
,
...
passages_path=dataset_path,
...
index_path=index_path,
...
)
>>>
# To load the legacy index built originally for Rag's paper
>>>
from
transformers
import
RagRetriever
>>>
retriever = RagRetriever.from_pretrained(
"facebook/rag-sequence-nq"
, index_name=
"legacy"
)
init_retrieval
<
source
>
(
)
Retriever initialization function. It loads the index into memory.
postprocess_docs
<
source
>
(
docs
input_strings
prefix
n_docs
return_tensors
= None
)
→
tuple(tensors)
Parameters
docs
(
dict
) —
Retrieved documents.
input_strings
(
str
) —
Input strings decoded by
preprocess_query
.
prefix
(
str
) —
Prefix added at the beginning of each input, typically used with T5-based models.
Returns
tuple(tensors)
a tuple consisting of two elements: contextualized
input_ids
and a compatible
attention_mask
.
Postprocessing retrieved
docs
and combining them with
input_strings
.
retrieve
<
source
>
(
question_hidden_states
: ndarray
n_docs
: int
)
→
tuple[np.ndarray, np.ndarray, list[dict]]
Parameters
question_hidden_states
(
np.ndarray
of shape
(batch_size, vector_size)
) —
A batch of query vectors to retrieve with.
n_docs
(
int
) —
The number of docs retrieved per query.
Returns
tuple[np.ndarray, np.ndarray, list[dict]]
A tuple with the following objects:
retrieved_doc_embeds
(
np.ndarray
of shape
(batch_size, n_docs, dim)
) — The retrieval embeddings
of the retrieved docs per query.
doc_ids
(
np.ndarray
of shape
(batch_size, n_docs)
) — The ids of the documents in the index
doc_dicts
(
list[dict]
): The
retrieved_doc_embeds
examples per query.
Retrieves documents for specified
question_hidden_states
.
RagModel
class
transformers.
RagModel
<
source
>
(
config
: transformers.configuration_utils.PreTrainedConfig | None = None
question_encoder
: transformers.modeling_utils.PreTrainedModel | None = None
generator
: transformers.modeling_utils.PreTrainedModel | None = None
retriever
: transformers.models.rag.retrieval_rag.RagRetriever | None = None
**kwargs
)
Parameters
config
(
PreTrainedConfig
,
optional
) —
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
from_pretrained()
method to load the model weights.
question_encoder
(
PreTrainedModel
,
optional
) —
The model responsible for encoding the question into hidden states for retrieval.
generator
(
PreTrainedModel
,
optional
) —
The model responsible for generating text based on retrieved documents.
retriever
(
RagRetriever
,
optional
) —
The component responsible for retrieving documents from a knowledge base given the encoded question.
The bare Rag Model outputting raw hidden-states without any specific head on top.
This model inherits from
PreTrainedModel
. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)
This model is also a PyTorch
torch.nn.Module
subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.
forward
<
source
>
(
input_ids
: typing.Optional[torch.LongTensor] = None
attention_mask
: typing.Optional[torch.Tensor] = None
encoder_outputs
: tuple[tuple[torch.FloatTensor]] | None = None
decoder_input_ids
: typing.Optional[torch.LongTensor] = None
decoder_attention_mask
: typing.Optional[torch.BoolTensor] = None
past_key_values
: transformers.cache_utils.Cache | None = None
doc_scores
: typing.Optional[torch.FloatTensor] = None
context_input_ids
: typing.Optional[torch.LongTensor] = None
context_attention_mask
: typing.Optional[torch.LongTensor] = None
use_cache
: bool | None = None
output_attentions
: bool | None = None
output_hidden_states
: bool | None = None
output_retrieved
: bool | None = None
n_docs
: int | None = None
**kwargs
)
→
RetrievAugLMOutput
or
tuple(torch.FloatTensor)
Parameters
input_ids
(
torch.LongTensor
of shape
(batch_size, sequence_length)
) —
Indices of input sequence tokens in the vocabulary.
RagConfig
, used to initialize the model, specifies
which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
obtain the indices.
What are input IDs?
attention_mask
(
torch.Tensor
of shape
(batch_size, sequence_length)
,
optional
) —
Mask to avoid performing attention on padding token indices. Mask values selected in
[0, 1]
:
1 for tokens that are
not masked
,
0 for tokens that are
masked
.
What are attention masks?
encoder_outputs
(
tuple(tuple(torch.FloatTensor)
,
optional
) —
Tuple consists of (
generator_enc_last_hidden_state
,
optional
:
generator_enc_hidden_states
,
optional
:
generator_enc_attentions
).
generator_enc_last_hidden_state
of shape
(batch_size, n_docs * sequence_length, hidden_size)
is a sequence of hidden-states at the output of the last layer of the
generator’s encoder.
Used by the (
RagModel
) model during decoding.
decoder_input_ids
(
torch.LongTensor
of shape
(batch_size, target_sequence_length)
,
optional
) —
Provide for generation tasks.
None
by default, construct as per instructions for the generator model
you’re using with your RAG instance.
decoder_input_ids
(
torch.LongTensor
of shape
(batch_size, target_sequence_length)
,
optional
) —
Indices of decoder input sequence tokens in the vocabulary.
Indices can be obtained using
AutoTokenizer
. See
PreTrainedTokenizer.encode()
and
PreTrainedTokenizer.
call
()
for details.
What are decoder input IDs?
decoder_attention_mask
(
torch.BoolTensor
of shape
(batch_size, target_sequence_length)
,
optional
) —
Default behavior: generate a tensor that ignores pad tokens in
decoder_input_ids
. Causal mask will also
be used by default.
past_key_values
(
~cache_utils.Cache
,
optional
) —
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the
past_key_values
returned by the model at a previous stage of decoding, when
use_cache=True
or
config.use_cache=True
.
Only
Cache
instance is allowed as input, see our
kv cache guide
.
If no
past_key_values
are passed,
DynamicCache
will be initialized by default.
The model will output the same cache format that is fed as input.
If
past_key_values
are used, the user is expected to input only unprocessed
input_ids
(those that don’t
have their past key value states given to this model) of shape
(batch_size, unprocessed_length)
instead of all
input_ids
of shape
(batch_size, sequence_length)
.
doc_scores
(
torch.FloatTensor
of shape
(batch_size, config.n_docs)
) —
Score between each retrieved document embeddings (see
retrieved_doc_embeds
) and
question_encoder_last_hidden_state
. If the model has is not initialized with a
retriever
doc_scores
has to be provided to the forward pass.
doc_scores
can be computed via
question_encoder_last_hidden_state
and
retrieved_doc_embeds
, see examples for more information.
context_input_ids
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Input IDs post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever. If the model was not initialized with a
retriever
`
context_input_ids
has to be provided to
the forward pass.
context_input_ids
are returned by
__call__()
.
context_attention_mask
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Attention mask post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever. If the model has is not initialized with a
retriever
context_attention_mask
has to be
provided to the forward pass.
context_attention_mask
are returned by
__call__()
.
use_cache
(
bool
,
optional
) —
If set to
True
,
past_key_values
key value states are returned and can be used to speed up decoding (see
past_key_values
).
output_attentions
(
bool
,
optional
) —
Whether or not to return the attentions tensors of all attention layers. See
attentions
under returned
tensors for more detail.
output_hidden_states
(
bool
,
optional
) —
Whether or not to return the hidden states of all layers. See
hidden_states
under returned tensors for
more detail.
output_retrieved
(
bool
,
optional
) —
Whether or not to return the
retrieved_doc_embeds
,
retrieved_doc_ids
,
context_input_ids
and
context_attention_mask
. See returned tensors for more detail.
n_docs
(
int
,
optional
) —
The number of documents to retrieve.
Returns
RetrievAugLMOutput
or
tuple(torch.FloatTensor)
A
RetrievAugLMOutput
or a tuple of
torch.FloatTensor
(if
return_dict=False
is passed or when
config.return_dict=False
) comprising various
elements depending on the configuration (
RagConfig
) and inputs.
The
RagModel
forward method, overrides the
__call__
special method.
Although the recipe for forward pass needs to be defined within this function, one should call the
Module
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.
logits
(
torch.FloatTensor
of shape
(batch_size, sequence_length, config.vocab_size)
) — Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
each vocabulary token.
doc_scores
(
torch.FloatTensor
of shape
(batch_size, config.n_docs)
) — Score between each retrieved document embeddings (see
retrieved_doc_embeds
) and
question_encoder_last_hidden_state
.
past_key_values
(
Cache
,
optional
, returned when
use_cache=True
is passed or when
config.use_cache=True
) — It is a
Cache
instance. For more details, see our
kv cache guide
.
Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
(see
past_key_values
input) to speed up sequential decoding.
retrieved_doc_embeds
(
torch.FloatTensor
of shape
(batch_size, config.n_docs, hidden_size)
,
optional
, returned when
output_retrieved=True
) — Embedded documents retrieved by the retriever. Is used with
question_encoder_last_hidden_state
to compute
the
doc_scores
.
retrieved_doc_ids
(
torch.LongTensor
of shape
(batch_size, config.n_docs)
,
optional
, returned when
output_retrieved=True
) — The indexes of the embedded documents retrieved by the retriever.
context_input_ids
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) — Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
context_attention_mask
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) — Attention mask post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever.
question_encoder_last_hidden_state
(
torch.FloatTensor
of shape
(batch_size, sequence_length, hidden_size)
,
optional
) — Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
model.
question_enc_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) — Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
question_enc_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_enc_last_hidden_state
(
torch.FloatTensor
of shape
(batch_size, sequence_length, hidden_size)
,
optional
) — Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
generator_enc_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) — Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
generator_enc_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_dec_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) — Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
generator_dec_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_cross_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
weighted average in the cross-attention heads.
Example:
Copied
>>>
from
transformers
import
AutoTokenizer, RagRetriever, RagModel
>>>
import
torch
>>>
tokenizer = AutoTokenizer.from_pretrained(
"facebook/rag-token-base"
)
>>>
retriever = RagRetriever.from_pretrained(
...
"facebook/rag-token-base"
, index_name=
"exact"
, use_dummy_dataset=
True
...
)
>>>
# initialize with RagRetriever to do everything in one forward call
>>>
model = RagModel.from_pretrained(
"facebook/rag-token-base"
, retriever=retriever)
>>>
inputs = tokenizer(
"How many people live in Paris?"
, return_tensors=
"pt"
)
>>>
outputs = model(input_ids=inputs[
"input_ids"
])
RagSequenceForGeneration
class
transformers.
RagSequenceForGeneration
<
source
>
(
config
: transformers.configuration_utils.PreTrainedConfig | None = None
question_encoder
: transformers.modeling_utils.PreTrainedModel | None = None
generator
: transformers.modeling_utils.PreTrainedModel | None = None
retriever
: transformers.models.rag.retrieval_rag.RagRetriever | None = None
**kwargs
)
Parameters
config
(
PreTrainedConfig
,
optional
) —
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
from_pretrained()
method to load the model weights.
question_encoder
(
PreTrainedModel
,
optional
) —
The model responsible for encoding the question into hidden states for retrieval.
generator
(
PreTrainedModel
,
optional
) —
The model responsible for generating text based on retrieved documents.
retriever
(
RagRetriever
,
optional
) —
The component responsible for retrieving documents from a knowledge base given the encoded question.
A RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.
This model inherits from
PreTrainedModel
. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)
This model is also a PyTorch
torch.nn.Module
subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.
forward
<
source
>
(
input_ids
: typing.Optional[torch.LongTensor] = None
attention_mask
: typing.Optional[torch.Tensor] = None
encoder_outputs
: tuple[tuple[torch.Tensor]] | None = None
decoder_input_ids
: typing.Optional[torch.LongTensor] = None
decoder_attention_mask
: typing.Optional[torch.BoolTensor] = None
past_key_values
: transformers.cache_utils.Cache | None = None
context_input_ids
: typing.Optional[torch.LongTensor] = None
context_attention_mask
: typing.Optional[torch.LongTensor] = None
doc_scores
: typing.Optional[torch.FloatTensor] = None
use_cache
: bool | None = None
output_attentions
: bool | None = None
output_hidden_states
: bool | None = None
output_retrieved
: bool | None = None
exclude_bos_score
: bool | None = None
reduce_loss
: bool | None = None
labels
: typing.Optional[torch.LongTensor] = None
n_docs
: int | None = None
**kwargs
)
→
RetrievAugLMMarginOutput
or
tuple(torch.FloatTensor)
Parameters
input_ids
(
torch.LongTensor
of shape
(batch_size, sequence_length)
) —
Indices of input sequence tokens in the vocabulary.
RagConfig
, used to initialize the model, specifies
which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
obtain the indices.
What are input IDs?
attention_mask
(
torch.Tensor
of shape
(batch_size, sequence_length)
,
optional
) —
Mask to avoid performing attention on padding token indices. Mask values selected in
[0, 1]
:
1 for tokens that are
not masked
,
0 for tokens that are
masked
.
What are attention masks?
encoder_outputs
(
tuple(tuple(torch.FloatTensor)
,
optional
) —
Tuple consists of (
generator_enc_last_hidden_state
,
optional
:
generator_enc_hidden_states
,
optional
:
generator_enc_attentions
).
generator_enc_last_hidden_state
of shape
(batch_size, n_docs * sequence_length, hidden_size)
is a sequence of hidden-states at the output of the last layer of the
generator’s encoder.
Used by the (
RagModel
) model during decoding.
decoder_input_ids
(
torch.LongTensor
of shape
(batch_size, target_sequence_length)
,
optional
) —
Provide for generation tasks.
None
by default, construct as per instructions for the generator model
you’re using with your RAG instance.
decoder_input_ids
(
torch.LongTensor
of shape
(batch_size, target_sequence_length)
,
optional
) —
Indices of decoder input sequence tokens in the vocabulary.
Indices can be obtained using
AutoTokenizer
. See
PreTrainedTokenizer.encode()
and
PreTrainedTokenizer.
call
()
for details.
What are decoder input IDs?
decoder_attention_mask
(
torch.BoolTensor
of shape
(batch_size, target_sequence_length)
,
optional
) —
Default behavior: generate a tensor that ignores pad tokens in
decoder_input_ids
. Causal mask will also
be used by default.
past_key_values
(
~cache_utils.Cache
,
optional
) —
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the
past_key_values
returned by the model at a previous stage of decoding, when
use_cache=True
or
config.use_cache=True
.
Only
Cache
instance is allowed as input, see our
kv cache guide
.
If no
past_key_values
are passed,
DynamicCache
will be initialized by default.
The model will output the same cache format that is fed as input.
If
past_key_values
are used, the user is expected to input only unprocessed
input_ids
(those that don’t
have their past key value states given to this model) of shape
(batch_size, unprocessed_length)
instead of all
input_ids
of shape
(batch_size, sequence_length)
.
context_input_ids
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Input IDs post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever. If the model was not initialized with a
retriever
`
context_input_ids
has to be provided to
the forward pass.
context_input_ids
are returned by
__call__()
.
context_attention_mask
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Attention mask post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever. If the model has is not initialized with a
retriever
context_attention_mask
has to be
provided to the forward pass.
context_attention_mask
are returned by
__call__()
.
doc_scores
(
torch.FloatTensor
of shape
(batch_size, config.n_docs)
) —
Score between each retrieved document embeddings (see
retrieved_doc_embeds
) and
question_encoder_last_hidden_state
. If the model has is not initialized with a
retriever
doc_scores
has to be provided to the forward pass.
doc_scores
can be computed via
question_encoder_last_hidden_state
and
retrieved_doc_embeds
, see examples for more information.
use_cache
(
bool
,
optional
) —
If set to
True
,
past_key_values
key value states are returned and can be used to speed up decoding (see
past_key_values
).
output_attentions
(
bool
,
optional
) —
Whether or not to return the attentions tensors of all attention layers. See
attentions
under returned
tensors for more detail.
output_hidden_states
(
bool
,
optional
) —
Whether or not to return the hidden states of all layers. See
hidden_states
under returned tensors for
more detail.
output_retrieved
(
bool
,
optional
) —
Whether or not to return the
retrieved_doc_embeds
,
retrieved_doc_ids
,
context_input_ids
and
context_attention_mask
. See returned tensors for more detail.
exclude_bos_score
(
bool
,
optional
) —
Only relevant if
labels
is passed. If
True
, the score of the BOS token is disregarded when computing
the loss.
reduce_loss
(
bool
,
optional
) —
Only relevant if
labels
is passed. If
True
, the NLL loss is reduced using the
torch.Tensor.sum
operation.
labels
(
torch.LongTensor
of shape
(batch_size, sequence_length)
,
optional
) —
Labels for computing the masked language modeling loss. Indices should either be in
[0, ..., config.vocab_size]
or -100 (see
input_ids
docstring). Tokens with indices set to
-100
are ignored
(masked), the loss is only computed for the tokens with labels in
[0, ..., config.vocab_size]
.
n_docs
(
int
,
optional
) —
The number of documents to retrieve.
Returns
RetrievAugLMMarginOutput
or
tuple(torch.FloatTensor)
A
RetrievAugLMMarginOutput
or a tuple of
torch.FloatTensor
(if
return_dict=False
is passed or when
config.return_dict=False
) comprising various
elements depending on the configuration (
RagConfig
) and inputs.
The
RagSequenceForGeneration
forward method, overrides the
__call__
special method.
Although the recipe for forward pass needs to be defined within this function, one should call the
Module
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.
loss
(
torch.FloatTensor
of shape
(1,)
,
optional
, returned when
labels
is provided) — Language modeling loss.
logits
(
torch.FloatTensor
of shape
(batch_size, sequence_length, config.vocab_size)
) — Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
each vocabulary token.
doc_scores
(
torch.FloatTensor
of shape
(batch_size, config.n_docs)
) — Score between each retrieved document embeddings (see
retrieved_doc_embeds
) and
question_encoder_last_hidden_state
.
past_key_values
(
Cache
,
optional
, returned when
use_cache=True
is passed or when
config.use_cache=True
) — It is a
Cache
instance. For more details, see our
kv cache guide
.
Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
(see
past_key_values
input) to speed up sequential decoding.
retrieved_doc_embeds
(
torch.FloatTensor
of shape
(batch_size, config.n_docs, hidden_size)
,
optional
, returned when
output_retrieved=True
) — Embedded documents retrieved by the retriever. Is used with
question_encoder_last_hidden_state
to compute
the
doc_scores
.
retrieved_doc_ids
(
torch.LongTensor
of shape
(batch_size, config.n_docs)
,
optional
, returned when
output_retrieved=True
) — The indexes of the embedded documents retrieved by the retriever.
context_input_ids
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) — Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
context_attention_mask
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) — Attention mask post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever.
question_encoder_last_hidden_state
(
torch.FloatTensor
of shape
(batch_size, sequence_length, hidden_size)
,
optional
) — Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
model.
question_enc_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) — Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
question_enc_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_enc_last_hidden_state
(
torch.FloatTensor
of shape
(batch_size, sequence_length, hidden_size)
,
optional
) — Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
generator_enc_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) — Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
generator_enc_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_dec_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) — Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
generator_dec_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_cross_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
weighted average in the cross-attention heads.
Example:
Copied
>>>
from
transformers
import
AutoTokenizer, RagRetriever, RagSequenceForGeneration
>>>
import
torch
>>>
tokenizer = AutoTokenizer.from_pretrained(
"facebook/rag-sequence-nq"
)
>>>
retriever = RagRetriever.from_pretrained(
...
"facebook/rag-sequence-nq"
, index_name=
"exact"
, use_dummy_dataset=
True
...
)
>>>
# initialize with RagRetriever to do everything in one forward call
>>>
model = RagSequenceForGeneration.from_pretrained(
"facebook/rag-token-nq"
, retriever=retriever)
>>>
inputs = tokenizer(
"How many people live in Paris?"
, return_tensors=
"pt"
)
>>>
targets = tokenizer(text_target=
"In Paris, there are 10 million people."
, return_tensors=
"pt"
)
>>>
input_ids = inputs[
"input_ids"
]
>>>
labels = targets[
"input_ids"
]
>>>
outputs = model(input_ids=input_ids, labels=labels)
>>>
# or use retriever separately
>>>
model = RagSequenceForGeneration.from_pretrained(
"facebook/rag-sequence-nq"
, use_dummy_dataset=
True
)
>>>
# 1. Encode
>>>
question_hidden_states = model.question_encoder(input_ids)[
0
]
>>>
# 2. Retrieve
>>>
docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors=
"pt"
)
>>>
doc_scores = torch.bmm(
...
question_hidden_states.unsqueeze(
1
), docs_dict[
"retrieved_doc_embeds"
].
float
().transpose(
1
,
2
)
...
).squeeze(
1
)
>>>
# 3. Forward to generator
>>>
outputs = model(
...
context_input_ids=docs_dict[
"context_input_ids"
],
...
context_attention_mask=docs_dict[
"context_attention_mask"
],
...
doc_scores=doc_scores,
...
decoder_input_ids=labels,
...
)
generate
<
source
>
(
input_ids
: typing.Optional[torch.LongTensor] = None
attention_mask
: typing.Optional[torch.LongTensor] = None
context_input_ids
: typing.Optional[torch.LongTensor] = None
context_attention_mask
: typing.Optional[torch.LongTensor] = None
doc_scores
: typing.Optional[torch.FloatTensor] = None
do_deduplication
: bool | None = None
num_return_sequences
: int | None = None
num_beams
: int | None = None
n_docs
: int | None = None
**model_kwargs
)
→
torch.LongTensor
of shape
(batch_size * num_return_sequences, sequence_length)
Parameters
input_ids
(
torch.LongTensor
of shape
(batch_size, sequence_length)
,
optional
) —
The sequence used as a prompt for the generation. If
input_ids
is not passed, then
context_input_ids
has to be provided.
attention_mask
(
torch.Tensor
of shape
(batch_size, sequence_length)
,
optional
) —
Mask to avoid performing attention on padding token indices. Mask values selected in
[0, 1]
:
1 for tokens that are
not masked
,
0 for tokens that are
masked
.
What are attention masks?
context_input_ids
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Input IDs post-processed from the retrieved documents and the question encoder input_ids by the
retriever.
context_attention_mask
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Attention mask post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever.
If the model is not initialized with a
retriever
or
input_ids
is not given,
context_input_ids
and
context_attention_mask
have to be provided to the forward pass. They are returned by
__call__()
.
doc_scores
(
torch.FloatTensor
of shape
(batch_size, config.n_docs)
) —
Score between each retrieved document embeddings (see
retrieved_doc_embeds
) and
question_encoder_last_hidden_state
.
If the model is not initialized with a
retriever
or
input_ids
is not given,
doc_scores
has to be
provided to the forward pass.
doc_scores
are returned by
__call__()
.
do_deduplication
(
bool
,
optional
) —
Whether or not to deduplicate the generations from different context documents for a given input. Has
to be set to
False
if used while training with distributed backend.
num_return_sequences(
int
,
optional
, defaults to 1) —
The number of independently computed returned sequences for each element in the batch. Note that this
is not the value we pass to the
generator
’s
[generate()](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate)
function,
where we set
num_return_sequences
to
num_beams
.
num_beams
(
int
,
optional
, defaults to 1) —
Number of beams for beam search. 1 means no beam search.
n_docs
(
int
,
optional
, defaults to
config.n_docs
) —
Number of documents to retrieve and/or number of documents for which to generate an answer.
kwargs
(
dict[str, Any]
,
optional
) —
Additional kwargs will be passed to
generate()
.
Returns
torch.LongTensor
of shape
(batch_size * num_return_sequences, sequence_length)
The generated
sequences. The second dimension (sequence length) is either equal to
max_length
or shorter if all batches
finished early due to the
eos_token_id
.
Implements RAG sequence “thorough” decoding. Read the
generate()
` documentation
for more information on how to set other generate input parameters.
RagTokenForGeneration
class
transformers.
RagTokenForGeneration
<
source
>
(
config
: transformers.configuration_utils.PreTrainedConfig | None = None
question_encoder
: transformers.modeling_utils.PreTrainedModel | None = None
generator
: transformers.modeling_utils.PreTrainedModel | None = None
retriever
: transformers.models.rag.retrieval_rag.RagRetriever | None = None
**kwargs
)
Parameters
config
(
PreTrainedConfig
,
optional
) —
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
from_pretrained()
method to load the model weights.
question_encoder
(
PreTrainedModel
,
optional
) —
The model responsible for encoding the question into hidden states for retrieval.
generator
(
PreTrainedModel
,
optional
) —
The model responsible for generating text based on retrieved documents.
retriever
(
RagRetriever
,
optional
) —
The component responsible for retrieving documents from a knowledge base given the encoded question.
A RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.
This model inherits from
PreTrainedModel
. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)
This model is also a PyTorch
torch.nn.Module
subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.
forward
<
source
>
(
input_ids
: typing.Optional[torch.LongTensor] = None
attention_mask
: typing.Optional[torch.FloatTensor] = None
encoder_outputs
: tuple[tuple[torch.Tensor]] | None = None
decoder_input_ids
: typing.Optional[torch.LongTensor] = None
decoder_attention_mask
: typing.Optional[torch.BoolTensor] = None
past_key_values
: transformers.cache_utils.Cache | None = None
context_input_ids
: typing.Optional[torch.LongTensor] = None
context_attention_mask
: typing.Optional[torch.LongTensor] = None
doc_scores
: typing.Optional[torch.FloatTensor] = None
use_cache
: bool | None = None
output_attentions
: bool | None = None
output_hidden_states
: bool | None = None
output_retrieved
: bool | None = None
do_marginalize
: bool | None = None
reduce_loss
: bool | None = None
labels
: typing.Optional[torch.LongTensor] = None
n_docs
: int | None = None
**kwargs
)
→
RetrievAugLMMarginOutput
or
tuple(torch.FloatTensor)
Parameters
input_ids
(
torch.LongTensor
of shape
(batch_size, sequence_length)
) —
Indices of input sequence tokens in the vocabulary.
RagConfig
, used to initialize the model, specifies
which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
obtain the indices.
What are input IDs?
attention_mask
(
torch.FloatTensor
of shape
(batch_size, sequence_length)
,
optional
) —
Mask to avoid performing attention on padding token indices. Mask values selected in
[0, 1]
:
1 for tokens that are
not masked
,
0 for tokens that are
masked
.
What are attention masks?
encoder_outputs
(
tuple(tuple(torch.FloatTensor)
,
optional
) —
Tuple consists of (
generator_enc_last_hidden_state
,
optional
:
generator_enc_hidden_states
,
optional
:
generator_enc_attentions
).
generator_enc_last_hidden_state
of shape
(batch_size, n_docs * sequence_length, hidden_size)
is a sequence of hidden-states at the output of the last layer of the
generator’s encoder.
Used by the (
RagModel
) model during decoding.
decoder_input_ids
(
torch.LongTensor
of shape
(batch_size, target_sequence_length)
,
optional
) —
Provide for generation tasks.
None
by default, construct as per instructions for the generator model
you’re using with your RAG instance.
decoder_input_ids
(
torch.LongTensor
of shape
(batch_size, target_sequence_length)
,
optional
) —
Indices of decoder input sequence tokens in the vocabulary.
Indices can be obtained using
AutoTokenizer
. See
PreTrainedTokenizer.encode()
and
PreTrainedTokenizer.
call
()
for details.
What are decoder input IDs?
decoder_attention_mask
(
torch.BoolTensor
of shape
(batch_size, target_sequence_length)
,
optional
) —
Default behavior: generate a tensor that ignores pad tokens in
decoder_input_ids
. Causal mask will also
be used by default.
past_key_values
(
~cache_utils.Cache
,
optional
) —
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the
past_key_values
returned by the model at a previous stage of decoding, when
use_cache=True
or
config.use_cache=True
.
Only
Cache
instance is allowed as input, see our
kv cache guide
.
If no
past_key_values
are passed,
DynamicCache
will be initialized by default.
The model will output the same cache format that is fed as input.
If
past_key_values
are used, the user is expected to input only unprocessed
input_ids
(those that don’t
have their past key value states given to this model) of shape
(batch_size, unprocessed_length)
instead of all
input_ids
of shape
(batch_size, sequence_length)
.
context_input_ids
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Input IDs post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever. If the model was not initialized with a
retriever
`
context_input_ids
has to be provided to
the forward pass.
context_input_ids
are returned by
__call__()
.
context_attention_mask
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Attention mask post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever. If the model has is not initialized with a
retriever
context_attention_mask
has to be
provided to the forward pass.
context_attention_mask
are returned by
__call__()
.
doc_scores
(
torch.FloatTensor
of shape
(batch_size, config.n_docs)
) —
Score between each retrieved document embeddings (see
retrieved_doc_embeds
) and
question_encoder_last_hidden_state
. If the model has is not initialized with a
retriever
doc_scores
has to be provided to the forward pass.
doc_scores
can be computed via
question_encoder_last_hidden_state
and
retrieved_doc_embeds
, see examples for more information.
use_cache
(
bool
,
optional
) —
If set to
True
,
past_key_values
key value states are returned and can be used to speed up decoding (see
past_key_values
).
output_attentions
(
bool
,
optional
) —
Whether or not to return the attentions tensors of all attention layers. See
attentions
under returned
tensors for more detail.
output_hidden_states
(
bool
,
optional
) —
Whether or not to return the hidden states of all layers. See
hidden_states
under returned tensors for
more detail.
output_retrieved
(
bool
,
optional
) —
Whether or not to return the
retrieved_doc_embeds
,
retrieved_doc_ids
,
context_input_ids
and
context_attention_mask
. See returned tensors for more detail.
do_marginalize
(
bool
,
optional
) —
If
True
, the logits are marginalized over all documents by making use of
torch.nn.functional.log_softmax
.
reduce_loss
(
bool
,
optional
) —
Only relevant if
labels
is passed. If
True
, the NLL loss is reduced using the
torch.Tensor.sum
operation.
labels
(
torch.LongTensor
of shape
(batch_size, sequence_length)
,
optional
) —
Labels for computing the masked language modeling loss. Indices should either be in
[0, ..., config.vocab_size]
or -100 (see
input_ids
docstring). Tokens with indices set to
-100
are ignored
(masked), the loss is only computed for the tokens with labels in
[0, ..., config.vocab_size]
.
n_docs
(
int
,
optional
) —
The number of documents to retrieve.
Returns
RetrievAugLMMarginOutput
or
tuple(torch.FloatTensor)
A
RetrievAugLMMarginOutput
or a tuple of
torch.FloatTensor
(if
return_dict=False
is passed or when
config.return_dict=False
) comprising various
elements depending on the configuration (
RagConfig
) and inputs.
The
RagTokenForGeneration
forward method, overrides the
__call__
special method.
Although the recipe for forward pass needs to be defined within this function, one should call the
Module
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.
loss
(
torch.FloatTensor
of shape
(1,)
,
optional
, returned when
labels
is provided) — Language modeling loss.
logits
(
torch.FloatTensor
of shape
(batch_size, sequence_length, config.vocab_size)
) — Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
each vocabulary token.
doc_scores
(
torch.FloatTensor
of shape
(batch_size, config.n_docs)
) — Score between each retrieved document embeddings (see
retrieved_doc_embeds
) and
question_encoder_last_hidden_state
.
past_key_values
(
Cache
,
optional
, returned when
use_cache=True
is passed or when
config.use_cache=True
) — It is a
Cache
instance. For more details, see our
kv cache guide
.
Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
(see
past_key_values
input) to speed up sequential decoding.
retrieved_doc_embeds
(
torch.FloatTensor
of shape
(batch_size, config.n_docs, hidden_size)
,
optional
, returned when
output_retrieved=True
) — Embedded documents retrieved by the retriever. Is used with
question_encoder_last_hidden_state
to compute
the
doc_scores
.
retrieved_doc_ids
(
torch.LongTensor
of shape
(batch_size, config.n_docs)
,
optional
, returned when
output_retrieved=True
) — The indexes of the embedded documents retrieved by the retriever.
context_input_ids
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) — Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
context_attention_mask
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) — Attention mask post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever.
question_encoder_last_hidden_state
(
torch.FloatTensor
of shape
(batch_size, sequence_length, hidden_size)
,
optional
) — Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
model.
question_enc_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) — Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
question_enc_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_enc_last_hidden_state
(
torch.FloatTensor
of shape
(batch_size, sequence_length, hidden_size)
,
optional
) — Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
generator_enc_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) — Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
generator_enc_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_dec_hidden_states
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_hidden_states=True
is passed or when
config.output_hidden_states=True
) — Tuple of
torch.FloatTensor
(one for the output of the embeddings and one for the output of each layer) of
shape
(batch_size, sequence_length, hidden_size)
.
Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
generator_dec_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.
generator_cross_attentions
(
tuple(torch.FloatTensor)
,
optional
, returned when
output_attentions=True
is passed or when
config.output_attentions=True
) — Tuple of
torch.FloatTensor
(one for each layer) of shape
(batch_size, num_heads, sequence_length, sequence_length)
.
Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
weighted average in the cross-attention heads.
Example:
Copied
>>>
from
transformers
import
AutoTokenizer, RagRetriever, RagTokenForGeneration
>>>
import
torch
>>>
tokenizer = AutoTokenizer.from_pretrained(
"facebook/rag-token-nq"
)
>>>
retriever = RagRetriever.from_pretrained(
...
"facebook/rag-token-nq"
, index_name=
"exact"
, use_dummy_dataset=
True
...
)
>>>
# initialize with RagRetriever to do everything in one forward call
>>>
model = RagTokenForGeneration.from_pretrained(
"facebook/rag-token-nq"
, retriever=retriever)
>>>
inputs = tokenizer(
"How many people live in Paris?"
, return_tensors=
"pt"
)
>>>
targets = tokenizer(text_target=
"In Paris, there are 10 million people."
, return_tensors=
"pt"
)
>>>
input_ids = inputs[
"input_ids"
]
>>>
labels = targets[
"input_ids"
]
>>>
outputs = model(input_ids=input_ids, labels=labels)
>>>
# or use retriever separately
>>>
model = RagTokenForGeneration.from_pretrained(
"facebook/rag-token-nq"
, use_dummy_dataset=
True
)
>>>
# 1. Encode
>>>
question_hidden_states = model.question_encoder(input_ids)[
0
]
>>>
# 2. Retrieve
>>>
docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors=
"pt"
)
>>>
doc_scores = torch.bmm(
...
question_hidden_states.unsqueeze(
1
), docs_dict[
"retrieved_doc_embeds"
].
float
().transpose(
1
,
2
)
...
).squeeze(
1
)
>>>
# 3. Forward to generator
>>>
outputs = model(
...
context_input_ids=docs_dict[
"context_input_ids"
],
...
context_attention_mask=docs_dict[
"context_attention_mask"
],
...
doc_scores=doc_scores,
...
decoder_input_ids=labels,
...
)
>>>
# or directly generate
>>>
generated = model.generate(
...
context_input_ids=docs_dict[
"context_input_ids"
],
...
context_attention_mask=docs_dict[
"context_attention_mask"
],
...
doc_scores=doc_scores,
...
)
>>>
generated_string = tokenizer.batch_decode(generated, skip_special_tokens=
True
)
generate
<
source
>
(
input_ids
: typing.Optional[torch.LongTensor] = None
attention_mask
: typing.Optional[torch.LongTensor] = None
context_input_ids
: typing.Optional[torch.LongTensor] = None
context_attention_mask
: typing.Optional[torch.LongTensor] = None
doc_scores
: typing.Optional[torch.FloatTensor] = None
n_docs
: int | None = None
generation_config
: transformers.generation.configuration_utils.GenerationConfig | None = None
prefix_allowed_tokens_fn
: collections.abc.Callable[[int, torch.Tensor], list[int]] | None = None
logits_processor
: transformers.generation.logits_process.LogitsProcessorList | None = []
stopping_criteria
: transformers.generation.stopping_criteria.StoppingCriteriaList | None = []
**kwargs
)
→
torch.LongTensor
of shape
(batch_size * num_return_sequences, sequence_length)
Parameters
input_ids
(
torch.LongTensor
of shape
(batch_size, sequence_length)
,
optional
) —
The sequence used as a prompt for the generation. If
input_ids
is not passed, then
context_input_ids
has to be provided.
attention_mask
(
torch.Tensor
of shape
(batch_size, sequence_length)
,
optional
) —
Mask to avoid performing attention on padding token indices. Mask values selected in
[0, 1]
:
1 for tokens that are
not masked
,
0 for tokens that are
masked
.
What are attention masks?
context_input_ids
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Input IDs post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever.
If the model is not initialized with a
retriever
,
context_input_ids
has to be provided to the
forward pass.
context_input_ids
are returned by
__call__()
.
context_attention_mask
(
torch.LongTensor
of shape
(batch_size * config.n_docs, config.max_combined_length)
,
optional
, returned when
output_retrieved=True
) —
Attention mask post-processed from the retrieved documents and the question encoder
input_ids
by the
retriever.
If the model is not initialized with a
retriever
,
context_input_ids
has to be provided to the
forward pass.
context_input_ids
are returned by
__call__()
.
doc_scores
(
torch.FloatTensor
of shape
(batch_size, config.n_docs)
) —
Score between each retrieved document embeddings (see
retrieved_doc_embeds
) and
question_encoder_last_hidden_state
.
If the model is not initialized with a
retriever
,
context_input_ids
has to be provided to the
forward pass.
context_input_ids
are returned by
__call__()
.
n_docs
(
int
,
optional
, defaults to
config.n_docs
) —
Number of documents to retrieve and/or number of documents for which to generate an answer.
generation_config
(
~generation.GenerationConfig
,
optional
) —
The generation configuration to be used as base parametrization for the generation call.
**kwargs
passed to generate matching the attributes of
generation_config
will override them. If
generation_config
is not provided, the default will be used, which has the following loading
priority: 1) from the
generation_config.json
model file, if it exists; 2) from the model
configuration. Please note that unspecified parameters will inherit
GenerationConfig
’s
default values, whose documentation should be checked to parameterize generation.
prefix_allowed_tokens_fn
(
Callable[[int, torch.Tensor], list[int]]
,
optional
) —
If provided, this function constraints the beam search to allowed tokens only at each step. If not
provided no constraint is applied. This function takes 2 arguments
inputs_ids
and the batch ID
batch_id
. It has to return a list with the allowed tokens for the next generation step conditioned on
the previously generated tokens
inputs_ids
and the batch ID
batch_id
. This argument is useful for
constrained generation conditioned on the prefix, as described in
Autoregressive Entity
Retrieval
.
logits_processor
(
LogitsProcessorList
,
optional
) —
Custom logits processors that complement the default logits processors built from arguments and a
model’s config. If a logit processor is passed that is already created with the arguments or a model’s
config an error is thrown.
stopping_criteria
(
StoppingCriteriaList
,
optional
) —
Custom stopping criteria that complement the default stopping criteria built from arguments and a
model’s config. If a stopping criteria is passed that is already created with the arguments or a
model’s config an error is thrown.
kwargs
(
dict[str, Any]
,
optional
) —
Ad hoc parametrization of
generate_config
and/or additional model-specific kwargs that will be
forwarded to the
forward
function of the model.
Returns
torch.LongTensor
of shape
(batch_size * num_return_sequences, sequence_length)
The generated
sequences. The second dimension (sequence_length) is either equal to
max_length
or shorter if all batches
finished early due to the
eos_token_id
.
Implements RAG token decoding.
Update
on GitHub
←
Qwen4-Exp
RecurrentGemma
→


---


## Hugging Face RAG with Chat Templates, Tools and Documents

**Source:** https://huggingface.co/docs/transformers/main/chat_template_tools_and_documents

Transformers documentation
Expanding Chat Templates with Tools and Documents
Transformers
🏡 View all docs
AWS Trainium & Inferentia
Accelerate
Argilla
AutoTrain
Bitsandbytes
CLI
Chat UI
Dataset viewer
Datasets
Deploying on AWS
Diffusers
Distilabel
Evaluate
Google Cloud
Google TPUs
Gradio
Hub
Hub Python Library
Huggingface.js
Inference Endpoints (dedicated)
Inference Providers
Kernels
LeRobot
Leaderboards
Lighteval
Microsoft Azure
OpenEnv
Optimum
PEFT
Reachy Mini
Safetensors
Sentence Transformers
TRL
Tasks
Text Embeddings Inference
Text Generation Inference
Tokenizers
Trackio
Transformers
Transformers.js
Xet
smolagents
timm
Search documentation
main
v5.17.0
v5.15.1
v5.14.0
v5.13.1
v5.12.0
v5.11.0
v5.10.4
v5.9.0
v5.8.1
v5.7.0
v5.6.2
v5.5.4
v5.4.0
v5.3.0
v5.2.0
v5.1.0
v5.0.0
v4.57.6
v4.56.2
v4.55.4
v4.53.3
v4.52.3
v4.51.3
v4.50.0
v4.49.0
v4.48.2
v4.47.1
v4.46.3
v4.45.2
v4.44.2
v4.43.4
v4.42.4
v4.41.2
v4.40.2
v4.39.3
v4.38.2
v4.37.2
v4.36.1
v4.35.2
v4.34.1
v4.33.3
v4.32.1
v4.31.0
v4.30.0
v4.29.1
v4.28.1
v4.27.2
v4.26.1
v4.25.1
v4.24.0
v4.23.1
v4.22.2
v4.21.3
v4.20.1
v4.19.4
v4.18.0
v4.17.0
v4.16.2
v4.15.0
v4.14.1
v4.13.0
v4.12.5
v4.11.3
v4.10.1
v4.9.2
v4.8.2
v4.7.0
v4.6.0
v4.5.1
v4.4.2
v4.3.3
v4.2.2
v4.1.1
v4.0.1
v3.5.1
v3.4.0
v3.3.1
v3.2.0
v3.1.0
v3.0.2
v2.11.0
v2.10.0
v2.9.1
v2.8.0
v2.7.0
v2.6.0
v2.5.1
v2.4.1
v2.3.0
v2.2.2
v2.1.1
v2.0.0
v1.2.0
v1.1.0
v1.0.0
doc-builder-html
AR
DE
EN
ES
FR
HI
IT
JA
KO
PT
RO
TE
TR
ZH
You are viewing
main
version, which requires
installation from source
. If you'd like
 regular pip install, checkout the latest stable version (
v5.17.0
).
Join the Hugging Face community
and get access to the augmented documentation experience
Collaborate on models, datasets and Spaces
Faster examples with accelerated inference
Switch between documentation themes
Sign Up
to get started
Expanding Chat Templates with Tools and Documents
The only argument that
apply_chat_template
requires is
messages
. However, you can pass any keyword
argument to
apply_chat_template
and it will be accessible inside the template. This gives you a lot of freedom to use
chat templates for many things. There are no restrictions on the names or the format of these arguments - you can pass
strings, lists, dicts or whatever else you want.
That said, there are some common use-cases for these extra arguments,
such as passing tools for function calling, or documents for retrieval-augmented generation. In these common cases,
we have some opinionated recommendations about what the names and formats of these arguments should be, which are
described in the sections below. We encourage model authors to make their chat templates compatible with this format,
to make it easy to transfer tool-calling code between models.
Tool use / function calling
“Tool use” LLMs can choose to call functions as external tools before generating an answer. When passing tools
to a tool-use model, you can simply pass a list of functions to the
tools
argument:
Copied
import
datetime
def
current_time
():
"""Get the current local time as a string."""
return
str
(datetime.now())
def
multiply
(
a:
float
, b:
float
):
"""
 A function that multiplies two numbers

 Args:
 a: The first number to multiply
 b: The second number to multiply
 """
return
a * b

tools = [current_time, multiply]

model_input = tokenizer.apply_chat_template(
 messages,
 tools=tools
)
In order for this to work correctly, you should write your functions in the format above, so that they can be parsed
correctly as tools. Specifically, you should follow these rules:
The function should have a descriptive name
Every argument must have a type hint
The function must have a docstring in the standard Google style (in other words, an initial function description
followed by an
Args:
block that describes the arguments, unless the function does not have any arguments.)
Do not include types in the
Args:
block. In other words, write
a: The first number to multiply
, not
a (int): The first number to multiply
. Type hints should go in the function header instead.
The function can have a return type and a
Returns:
block in the docstring. However, these are optional
because most tool-use models ignore them.
Passing tool results to the model
The sample code above is enough to list the available tools for your model, but what happens if it wants to actually use
one? If that happens, you should:
Parse the model’s output to get the tool name(s) and arguments.
Add the model’s tool call(s) to the conversation.
Call the corresponding function(s) with those arguments.
Add the result(s) to the conversation
A complete tool use example
Let’s walk through a tool use example, step by step. For this example, we will use an 8B
Hermes-2-Pro
model,
as it is one of the highest-performing tool-use models in its size category at the time of writing. If you have the
memory, you can consider using a larger model instead like
Command-R
or
Mixtral-8x22B
, both of which also support tool use
and offer even stronger performance.
First, let’s load our model and tokenizer:
Copied
import
torch
from
transformers
import
AutoModelForCausalLM, AutoTokenizer

checkpoint =
"NousResearch/Hermes-2-Pro-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map=
"auto"
)
Next, let’s define a list of tools:
Copied
def
get_current_temperature
(
location:
str
, unit:
str
) ->
float
:
"""
 Get the current temperature at a location.

 Args:
 location: The location to get the temperature for, in the format "City, Country"
 unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
 Returns:
 The current temperature at the specified location in the specified units, as a float.
 """
return
22.
# A real function should probably actually get the temperature!
def
get_current_wind_speed
(
location:
str
) ->
float
:
"""
 Get the current wind speed in km/h at a given location.

 Args:
 location: The location to get the temperature for, in the format "City, Country"
 Returns:
 The current wind speed at the given location in km/h, as a float.
 """
return
6.
# A real function should probably actually get the wind speed!
tools = [get_current_temperature, get_current_wind_speed]
Now, let’s set up a conversation for our bot:
Copied
messages = [
 {
"role"
:
"system"
,
"content"
:
"You are a bot that responds to weather queries. You should reply with the unit used in the queried location."
},
 {
"role"
:
"user"
,
"content"
:
"Hey, what's the temperature in Paris right now?"
}
]
Now, let’s apply the chat template and generate a response:
Copied
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=
True
, return_dict=
True
, return_tensors=
"pt"
)
inputs = {k: v.to(model.device)
for
k, v
in
inputs.items()}
out = model.generate(**inputs, max_new_tokens=
128
)
print
(tokenizer.decode(out[
0
][
len
(inputs[
"input_ids"
][
0
]):]))
And we get:
Copied
<tool_call>
{"arguments": {"location": "Paris, France", "unit": "celsius"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
The model has called the function with valid arguments, in the format requested by the function docstring. It has
inferred that we’re most likely referring to the Paris in France, and it remembered that, as the home of SI units,
the temperature in France should certainly be displayed in Celsius.
The output format above is specific to the
Hermes-2-Pro
model we’re using in this example. Other models may emit different
tool call formats, and you may need to do some manual parsing at this step. For example,
Llama-3.1
models will emit
slightly different JSON, with
parameters
instead of
arguments
. Regardless of the format the model outputs, you
should add the tool call to the conversation in the format below, with
tool_calls
,
function
and
arguments
keys.
Next, let’s append the model’s tool call to the conversation.
Copied
tool_call = {
"name"
:
"get_current_temperature"
,
"arguments"
: {
"location"
:
"Paris, France"
,
"unit"
:
"celsius"
}}
messages.append({
"role"
:
"assistant"
,
"tool_calls"
: [{
"type"
:
"function"
,
"function"
: tool_call}]})
If you’re familiar with the OpenAI API, you should pay attention to an important difference here - the
tool_call
is
a dict, but in the OpenAI API it’s a JSON string. Passing a string may cause errors or strange model behaviour!
Now that we’ve added the tool call to the conversation, we can call the function and append the result to the
conversation. Since we’re just using a dummy function for this example that always returns 22.0, we can just append
that result directly.
Copied
messages.append({
"role"
:
"tool"
,
"name"
:
"get_current_temperature"
,
"content"
:
"22.0"
})
Some model architectures, notably Mistral/Mixtral, also require a
tool_call_id
here, which should be
9 randomly-generated alphanumeric characters, and assigned to the
id
key of the tool call
dictionary. The same key should also be assigned to the
tool_call_id
key of the tool response dictionary below, so
that tool calls can be matched to tool responses. So, for Mistral/Mixtral models, the code above would be:
Copied
tool_call_id =
"9Ae3bDc2F"
# Random ID, 9 alphanumeric characters
tool_call = {
"name"
:
"get_current_temperature"
,
"arguments"
: {
"location"
:
"Paris, France"
,
"unit"
:
"celsius"
}}
messages.append({
"role"
:
"assistant"
,
"tool_calls"
: [{
"type"
:
"function"
,
"id"
: tool_call_id,
"function"
: tool_call}]})
and
Copied
messages.append({
"role"
:
"tool"
,
"tool_call_id"
: tool_call_id,
"name"
:
"get_current_temperature"
,
"content"
:
"22.0"
})
Finally, let’s let the assistant read the function outputs and continue chatting with the user:
Copied
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=
True
, return_dict=
True
, return_tensors=
"pt"
)
inputs = {k: v.to(model.device)
for
k, v
in
inputs.items()}
out = model.generate(**inputs, max_new_tokens=
128
)
print
(tokenizer.decode(out[
0
][
len
(inputs[
"input_ids"
][
0
]):]))
And we get:
Copied
The current temperature in Paris, France is 22.0 ° Celsius.<|im_end|>
Although this was a simple demo with dummy tools and a single call, the same technique works with
multiple real tools and longer conversations. This can be a powerful way to extend the capabilities of conversational
agents with real-time information, computational tools like calculators, or access to large databases.
Understanding tool schemas
Each function you pass to the
tools
argument of
apply_chat_template
is converted into a
JSON schema
. These schemas
are then passed to the model chat template. In other words, tool-use models do not see your functions directly, and they
never see the actual code inside them. What they care about is the function
definitions
and the
arguments
they
need to pass to them - they care about what the tools do and how to use them, not how they work! It is up to you
to read their outputs, detect if they have requested to use a tool, pass their arguments to the tool function, and
return the response in the chat.
Generating JSON schemas to pass to the template should be automatic and invisible as long as your functions
follow the specification above, but if you encounter problems, or you simply want more control over the conversion,
you can handle the conversion manually. Here is an example of a manual schema conversion.
Copied
from
transformers.utils
import
get_json_schema
def
multiply
(
a:
float
, b:
float
):
"""
 A function that multiplies two numbers

 Args:
 a: The first number to multiply
 b: The second number to multiply
 """
return
a * b

schema = get_json_schema(multiply)
print
(schema)
This will yield:
Copied
{
"type"
:
"function"
,
"function"
:
{
"name"
:
"multiply"
,
"description"
:
"A function that multiplies two numbers"
,
"parameters"
:
{
"type"
:
"object"
,
"properties"
:
{
"a"
:
{
"type"
:
"number"
,
"description"
:
"The first number to multiply"
}
,
"b"
:
{
"type"
:
"number"
,
"description"
:
"The second number to multiply"
}
}
,
"required"
:
[
"a"
,
"b"
]
}
}
}
If you wish, you can edit these schemas, or even write them from scratch yourself without using
get_json_schema
at
all. JSON schemas can be passed directly to the
tools
argument of
apply_chat_template
- this gives you a lot of power to define precise schemas for more complex functions. Be careful,
though - the more complex your schemas, the more likely the model is to get confused when dealing with them! We
recommend simple function signatures where possible, keeping arguments (and especially complex, nested arguments)
to a minimum.
Here is an example of defining schemas by hand, and passing them directly to
apply_chat_template
:
Copied
# A simple function that takes no arguments
current_time = {
"type"
:
"function"
,
"function"
: {
"name"
:
"current_time"
,
"description"
:
"Get the current local time as a string."
,
"parameters"
: {
'type'
:
'object'
,
'properties'
: {}
 }
 }
}
# A more complete function that takes two numerical arguments
multiply = {
'type'
:
'function'
,
'function'
: {
'name'
:
'multiply'
,
'description'
:
'A function that multiplies two numbers'
,
'parameters'
: {
'type'
:
'object'
,
'properties'
: {
'a'
: {
'type'
:
'number'
,
'description'
:
'The first number to multiply'
},
'b'
: {
'type'
:
'number'
,
'description'
:
'The second number to multiply'
}
 },
'required'
: [
'a'
,
'b'
]
 }
 }
}

model_input = tokenizer.apply_chat_template(
 messages,
 tools = [current_time, multiply]
)
Retrieval-augmented generation
“Retrieval-augmented generation” or “RAG” LLMs can search a corpus of documents for information before responding
to a query. This allows models to vastly expand their knowledge base beyond their limited context size. Our
recommendation for RAG models is that their template
should accept a
documents
argument. This should be a list of documents, where each “document”
is a single dict with
title
and
contents
keys, both of which are strings. Because this format is much simpler
than the JSON schemas used for tools, no helper functions are necessary.
Here’s an example of a RAG template in action:
Copied
from
transformers
import
AutoTokenizer, AutoModelForCausalLM
# Load the model and tokenizer
model_id =
"CohereForAI/c4ai-command-r-v01-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=
"auto"
)
device = model.device
# Get the device the model is loaded on
# Define conversation input
conversation = [
 {
"role"
:
"user"
,
"content"
:
"What has Man always dreamed of?"
}
]
# Define documents for retrieval-based generation
documents = [
 {
"title"
:
"The Moon: Our Age-Old Foe"
,
"text"
:
"Man has always dreamed of destroying the moon. In this essay, I shall..."
},
 {
"title"
:
"The Sun: Our Age-Old Friend"
,
"text"
:
"Although often underappreciated, the sun provides several notable benefits..."
}
]
# Tokenize conversation and documents using a RAG template, returning PyTorch tensors.
input_ids = tokenizer.apply_chat_template(
 conversation=conversation,
 documents=documents,
 chat_template=
"rag"
,
 tokenize=
True
,
 add_generation_prompt=
True
,
 return_tensors=
"pt"
).to(device)
# Generate a response
gen_tokens = model.generate(
 input_ids,
 max_new_tokens=
100
,
 do_sample=
True
,
 temperature=
0.3
,
 )
# Decode and print the generated text along with generation prompt
gen_text = tokenizer.decode(gen_tokens[
0
])
print
(gen_text)
The
documents
input for retrieval-augmented generation is not widely supported, and many models have chat templates which simply ignore this input.
To verify if a model supports the
documents
input, you can read its model card, or
print(tokenizer.chat_template)
to see if the
documents
key is used anywhere.
One model class that does support it, though, is Cohere’s
Command-R
and
Command-R+
, through their
rag
chat template. You can see additional examples of grounded generation using this feature in their model cards.
<
>
Update
on GitHub
Transformers
→


---


## Hugging Face Advanced Chat Templates

**Source:** https://huggingface.co/docs/transformers/main/chat_template_advanced

Transformers documentation
Advanced Usage and Customizing Your Chat Templates
Transformers
🏡 View all docs
AWS Trainium & Inferentia
Accelerate
Argilla
AutoTrain
Bitsandbytes
CLI
Chat UI
Dataset viewer
Datasets
Deploying on AWS
Diffusers
Distilabel
Evaluate
Google Cloud
Google TPUs
Gradio
Hub
Hub Python Library
Huggingface.js
Inference Endpoints (dedicated)
Inference Providers
Kernels
LeRobot
Leaderboards
Lighteval
Microsoft Azure
OpenEnv
Optimum
PEFT
Reachy Mini
Safetensors
Sentence Transformers
TRL
Tasks
Text Embeddings Inference
Text Generation Inference
Tokenizers
Trackio
Transformers
Transformers.js
Xet
smolagents
timm
Search documentation
main
v5.17.0
v5.15.1
v5.14.0
v5.13.1
v5.12.0
v5.11.0
v5.10.4
v5.9.0
v5.8.1
v5.7.0
v5.6.2
v5.5.4
v5.4.0
v5.3.0
v5.2.0
v5.1.0
v5.0.0
v4.57.6
v4.56.2
v4.55.4
v4.53.3
v4.52.3
v4.51.3
v4.50.0
v4.49.0
v4.48.2
v4.47.1
v4.46.3
v4.45.2
v4.44.2
v4.43.4
v4.42.4
v4.41.2
v4.40.2
v4.39.3
v4.38.2
v4.37.2
v4.36.1
v4.35.2
v4.34.1
v4.33.3
v4.32.1
v4.31.0
v4.30.0
v4.29.1
v4.28.1
v4.27.2
v4.26.1
v4.25.1
v4.24.0
v4.23.1
v4.22.2
v4.21.3
v4.20.1
v4.19.4
v4.18.0
v4.17.0
v4.16.2
v4.15.0
v4.14.1
v4.13.0
v4.12.5
v4.11.3
v4.10.1
v4.9.2
v4.8.2
v4.7.0
v4.6.0
v4.5.1
v4.4.2
v4.3.3
v4.2.2
v4.1.1
v4.0.1
v3.5.1
v3.4.0
v3.3.1
v3.2.0
v3.1.0
v3.0.2
v2.11.0
v2.10.0
v2.9.1
v2.8.0
v2.7.0
v2.6.0
v2.5.1
v2.4.1
v2.3.0
v2.2.2
v2.1.1
v2.0.0
v1.2.0
v1.1.0
v1.0.0
doc-builder-html
AR
DE
EN
ES
FR
HI
IT
JA
KO
PT
RO
TE
TR
ZH
You are viewing
main
version, which requires
installation from source
. If you'd like
 regular pip install, checkout the latest stable version (
v5.17.0
).
Join the Hugging Face community
and get access to the augmented documentation experience
Collaborate on models, datasets and Spaces
Faster examples with accelerated inference
Switch between documentation themes
Sign Up
to get started
Advanced Usage and Customizing Your Chat Templates
In this page, we’ll explore more advanced techniques for working with chat templates in Transformers. Whether you’re looking to write your own templates, create custom components, or optimize your templates for efficiency, we’ll cover everything you need to take your templates to the next level. Let’s dive into the tools and strategies that will help you get the most out of your chat models.
How do chat templates work?
The chat template for a model is stored on the
tokenizer.chat_template
attribute. Let’s take a look at a
Zephyr
chat template, though note this
one is a little simplified from the actual one!
Copied
{%- for message in messages %}
{{
-
'<|'
+ message['role'] +
'|>\n'
}}
{{
-
message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
{{
-
'<|assistant|>\n'
}}
{%- endif %}
If you’ve never seen one of these before, this is a
Jinja template
.
Jinja is a templating language that allows you to write simple code that generates text. In many ways, the code and
syntax resembles Python. In pure Python, this template would look something like this:
Copied
for
message
in
messages:
print
(
f'<|
{message[
"role"
]}
|>'
)
print
(message[
'content'
] + eos_token)
if
add_generation_prompt:
print
(
'<|assistant|>'
)
Effectively, the template does three things:
For each message, print the role enclosed in
<|
and
|>
, like
<|user|>
or
<|assistant|>
.
Next, print the content of the message, followed by the end-of-sequence token.
Finally, if
add_generation_prompt
is set, print the assistant token, so that the model knows to start generating
an assistant response.
This is a pretty simple template but Jinja gives you a lot of flexibility to do more complex things! Let’s see a Jinja
template that can format inputs similarly to the way LLaMA formats them (note that the real LLaMA template includes
handling for default system messages and slightly different system message handling in general - don’t use this one
in your actual code!)
Copied
{%-
for
message
in
messages %}
 {%-
if
message[
'role'
] ==
'user'
%}
 {{- bos_token +
'[INST] '
+ message[
'content'
] +
' [/INST]'
}}
 {%- elif message[
'role'
] ==
'system'
%}
 {{-
'<<SYS>>\\n'
+ message[
'content'
] +
'\\n<</SYS>>\\n\\n'
}}
 {%- elif message[
'role'
] ==
'assistant'
%}
 {{-
' '
+ message[
'content'
] +
' '
+ eos_token }}
 {%- endif %}
{%- endfor %}
Hopefully if you stare at this for a little bit you can see what this template is doing - it adds specific tokens like
[INST]
and
[/INST]
based on the role of each message. User, assistant and system messages are clearly
distinguishable to the model because of the tokens they’re wrapped in.
How do I create a chat template?
Simple, just write a jinja template and set
tokenizer.chat_template
. You may find it easier to start with an
existing template from another model and simply edit it for your needs! For example, we could take the LLaMA template
above and add ”[ASST]” and ”[/ASST]” to assistant messages:
Copied
{%-
for
message
in
messages %}
 {%-
if
message[
'role'
] ==
'user'
%}
 {{- bos_token +
'[INST] '
+ message[
'content'
].
strip
() +
' [/INST]'
}}
 {%- elif message[
'role'
] ==
'system'
%}
 {{-
'<<SYS>>\\n'
+ message[
'content'
].
strip
() +
'\\n<</SYS>>\\n\\n'
}}
 {%- elif message[
'role'
] ==
'assistant'
%}
 {{-
'[ASST] '
+ message[
'content'
] +
' [/ASST]'
+ eos_token }}
 {%- endif %}
{%- endfor %}
Now, simply set the
tokenizer.chat_template
attribute. Next time you use
apply_chat_template()
, it will
use your new template! This attribute will be saved in the
tokenizer_config.json
file, so you can use
push_to_hub()
to upload your new template to the Hub and make sure everyone’s using the right
template for your model!
Copied
template = tokenizer.chat_template
template = template.replace(
"SYS"
,
"SYSTEM"
)
# Change the system token
tokenizer.chat_template = template
# Set the new template
tokenizer.push_to_hub(
"model_name"
)
# Upload your new template to the Hub!
The method
apply_chat_template()
which uses your chat template is called by the
TextGenerationPipeline
class, so
once you set the correct chat template, your model will automatically become compatible with
TextGenerationPipeline
.
If you're fine-tuning a model for chat, in addition to setting a chat template, you should probably add any new chat
control tokens as special tokens in the tokenizer. Special tokens are never split, 
ensuring that your control tokens are always handled as single tokens rather than being tokenized in pieces. You 
should also set the tokenizer's `eos_token` attribute to the token that marks the end of assistant generations in your
template. This will ensure that text generation tools can correctly figure out when to stop generating text.
Why do some models have multiple templates?
Some models use different templates for different use cases. For example, they might use one template for normal chat
and another for tool-use, or retrieval-augmented generation. In these cases,
tokenizer.chat_template
is a dictionary.
This can cause some confusion, and where possible, we recommend using a single template for all use-cases. You can use
Jinja statements like
if tools is defined
and
{% macro %}
definitions to easily wrap multiple code paths in a
single template.
When a tokenizer has multiple templates,
tokenizer.chat_template
will be a
dict
, where each key is the name
of a template. The
apply_chat_template
method has special handling for certain template names: Specifically, it will
look for a template named
default
in most cases, and will raise an error if it can’t find one. However, if a template
named
tool_use
exists when the user has passed a
tools
argument, it will use that instead. To access templates
with other names, pass the name of the template you want to the
chat_template
argument of
apply_chat_template()
.
We find that this can be a bit confusing for users, though - so if you’re writing a template yourself, we recommend
trying to put it all in a single template where possible!
What template should I use?
When setting the template for a model that’s already been trained for chat, you should ensure that the template
exactly matches the message formatting that the model saw during training, or else you will probably experience
performance degradation. This is true even if you’re training the model further - you will probably get the best
performance if you keep the chat tokens constant. This is very analogous to tokenization - you generally get the
best performance for inference or fine-tuning when you precisely match the tokenization used during training.
If you’re training a model from scratch, or fine-tuning a base language model for chat, on the other hand,
you have a lot of freedom to choose an appropriate template! LLMs are smart enough to learn to handle lots of different
input formats. One popular choice is the
ChatML
format, and this is a good, flexible choice for many use-cases.
It looks like this:
Copied
{%- for message in messages %}
{{
-
'<|im_start|>'
+ message['role'] +
'\n'
+ message['content'] +
'<|im_end|>'
+
'\n'
}}
{%- endfor %}
If you like this one, here it is in one-liner form, ready to copy into your code. The one-liner also includes
handy support for
generation prompts
, but note that it doesn’t add BOS or EOS tokens!
If your model expects those, they won’t be added automatically by
apply_chat_template
- in other words, the
text will be tokenized with
add_special_tokens=False
. This is to avoid potential conflicts between the template and
the
add_special_tokens
logic. If your model expects special tokens, make sure to add them to the template!
Copied
tokenizer.chat_template =
"{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
This template wraps each message in
<|im_start|>
and
<|im_end|>
tokens, and simply writes the role as a string, which
allows for flexibility in the roles you train with. The output looks like this:
Copied
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
The “user”, “system” and “assistant” roles are the standard for chat, and we recommend using them when it makes sense,
particularly if you want your model to operate well with
TextGenerationPipeline
. However, you are not limited
to these roles - templating is extremely flexible, and any string can be a role.
I want to add some chat templates! How should I get started?
If you have any chat models, you should set their
tokenizer.chat_template
attribute and test it using
apply_chat_template()
, then push the updated tokenizer to the Hub. This applies even if you’re
not the model owner - if you’re using a model with an empty chat template, or one that’s still using the default class
template, please open a
pull request
to the model repository so that this attribute can be set properly!
Once the attribute is set, that’s it, you’re done!
tokenizer.apply_chat_template
will now work correctly for that
model, which means it is also automatically supported in places like
TextGenerationPipeline
!
By ensuring that models have this attribute, we can make sure that the whole community gets to use the full power of
open-source models. Formatting mismatches have been haunting the field and silently harming performance for too long -
it’s time to put an end to them!
The easiest way to get started with writing Jinja templates is to take a look at some existing ones. You can use
print(tokenizer.chat_template)
for any chat model to see what template it’s using. In general, models that support tool use have
much more complex templates than other models - so when you’re just getting started, they’re probably a bad example
to learn from! You can also take a look at the
Jinja documentation
for details
of general Jinja formatting and syntax.
Jinja templates in
transformers
are identical to Jinja templates elsewhere. The main thing to know is that
the conversation history will be accessible inside your template as a variable called
messages
.
You will be able to access
messages
in your template just like you can in Python, which means you can loop over
it with
{% for message in messages %}
or access individual messages with
{{ messages[0] }}
, for example.
You can also use the following tips to write clean, efficient Jinja templates:
Trimming whitespace
By default, Jinja will print any whitespace that comes before or after a block. This can be a problem for chat
templates, which generally want to be very precise with whitespace! To avoid this, we strongly recommend writing
your templates like this:
Copied
{%- for message in messages %}
{{
-
message['role'] + message['content'] }}
{%- endfor %}
rather than like this:
Copied
{%
for
message
in
messages
%}
{{ message[
'role'
] + message[
'content'
] }}
{%
endfor
%}
Adding
-
will strip any whitespace that comes before the block. The second example looks innocent, but the newline
and indentation may end up being included in the output, which is probably not what you want!
Special variables
Inside your template, you will have access several special variables. The most important of these is
messages
,
which contains the chat history as a list of message dicts. However, there are several others. Not every
variable will be used in every template. The most common other variables are:
tools
contains a list of tools in JSON schema format. Will be
None
or undefined if no tools are passed.
documents
contains a list of documents in the format
{"title": "Title", "contents": "Contents"}
, used for retrieval-augmented generation. Will be
None
or undefined if no documents are passed.
add_generation_prompt
is a bool that is
True
if the user has requested a generation prompt, and
False
otherwise. If this is set, your template should add the header for an assistant message to the end of the conversation. If your model doesn’t have a specific header for assistant messages, you can ignore this flag.
Special tokens
like
bos_token
and
eos_token
. These are extracted from
tokenizer.special_tokens_map
. The exact tokens available inside each template will differ depending on the parent tokenizer.
You can actually pass any
kwarg
to
apply_chat_template
, and it will be accessible inside the template as a variable. In general,
we recommend trying to stick to the core variables above, as it will make your model harder to use if users have
to write custom code to pass model-specific
kwargs
. However, we’re aware that this field moves quickly, so if you
have a new use-case that doesn’t fit in the core API, feel free to use a new
kwarg
for it! If a new
kwarg
becomes common we may promote it into the core API and create a standard, documented format for it.
Callable functions
There is also a short list of callable functions available to you inside your templates. These are:
raise_exception(msg)
: Raises a
TemplateException
. This is useful for debugging, and for telling users when they’re
doing something that your template doesn’t support.
strftime_now(format_str)
: Equivalent to
datetime.now().strftime(format_str)
in Python. This is used for getting
the current date/time in a specific format, which is sometimes included in system messages.
Compatibility with non-Python Jinja
There are multiple implementations of Jinja in various languages. They generally have the same syntax,
but a key difference is that when you’re writing a template in Python you can use Python methods, such as
.lower()
on strings or
.items()
on dicts. This will break if someone tries to use your template on a non-Python
implementation of Jinja. Non-Python implementations are particularly common in deployment environments, where JS
and Rust are very popular.
Don’t panic, though! There are a few easy changes you can make to your templates to ensure they’re compatible across
all implementations of Jinja:
Replace Python methods with Jinja filters. These usually have the same name, for example
string.lower()
becomes
string|lower
, and
dict.items()
becomes
dict|items
. One notable change is that
string.strip()
becomes
string|trim
.
See the
list of built-in filters
in the Jinja documentation for more.
Replace
True
,
False
and
None
, which are Python-specific, with
true
,
false
and
none
.
Directly rendering a dict or list may give different results in other implementations (for example, string entries
might change from single-quoted to double-quoted). Adding the
tojson
filter can help to ensure consistency here.
Writing generation prompts
We mentioned above that
add_generation_prompt
is a special variable that will be accessible inside your template,
and is controlled by the user setting the
add_generation_prompt
flag. If your model expects a header for
assistant messages, then your template must support adding the header when
add_generation_prompt
is set.
Here is an example of a template that formats messages ChatML-style, with generation prompt support:
Copied
{{- bos_token }}
{%- for message in messages %}
 {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
 {{- '<|im_start|>assistant\n' }}
{%- endif %}
The exact content of the assistant header will depend on your specific model, but it should always be
the string
that represents the start of an assistant message
, so that if the user applies your template with
add_generation_prompt=True
and then generates text, the model will write an assistant response. Also note that some
models do not need a generation prompt, because assistant messages always begin immediately after user messages.
This is particularly common for LLaMA and Mistral models, where assistant messages begin immediately after the
[/INST]
token that ends user messages. In these cases, the template can ignore the
add_generation_prompt
flag.
Generation prompts are important! If your model requires a generation prompt but it is not set in the template, then
model generations will likely be severely degraded, or the model may display unusual behaviour like continuing
the final user message!
Writing and debugging larger templates
When this feature was introduced, most templates were quite small, the Jinja equivalent of a “one-liner” script.
However, with new models and features like tool-use and RAG, some templates can be 100 lines long or more. When
writing templates like these, it’s a good idea to write them in a separate file, using a text editor. You can easily
extract a chat template to a file:
Copied
open
(
"template.jinja"
,
"w"
).write(tokenizer.chat_template)
Or load the edited template back into the tokenizer:
Copied
tokenizer.chat_template =
open
(
"template.jinja"
).read()
As an added bonus, when you write a long, multi-line template in a separate file, line numbers in that file will
exactly correspond to line numbers in template parsing or execution errors. This will make it much easier to
identify the source of issues.
Writing templates for tools
Although chat templates do not enforce a specific API for tools (or for anything, really), we recommend
template authors try to stick to a standard API where possible. The whole point of chat templates is to allow code
to be transferable across models, so deviating from the standard tools API means users will have to write
custom code to use tools with your model. Sometimes it’s unavoidable, but often with clever templating you can
make the standard API work!
Below, we’ll list the elements of the standard API, and give tips on writing templates that will work well with it.
Tool definitions
Your template should expect that the variable
tools
will either be null (if no tools are passed), or is a list
of JSON schema dicts. Our chat template methods allow users to pass tools as either JSON schema or Python functions, but when
functions are passed, we automatically generate JSON schema and pass that to your template. As a result, the
tools
variable that your template receives will always be a list of JSON schema. Here is
a sample tool JSON schema:
Copied
{
"type"
:
"function"
,
"function"
:
{
"name"
:
"multiply"
,
"description"
:
"A function that multiplies two numbers"
,
"parameters"
:
{
"type"
:
"object"
,
"properties"
:
{
"a"
:
{
"type"
:
"number"
,
"description"
:
"The first number to multiply"
}
,
"b"
:
{
"type"
:
"number"
,
"description"
:
"The second number to multiply"
}
}
,
"required"
:
[
"a"
,
"b"
]
}
}
}
And here is some example code for handling tools in your chat template. Remember, this is just an example for a
specific format - your model will probably need different formatting!
Copied
{%- if tools %}
 {%- for tool in tools %}
 {{- '<tool>' + tool['function']['name'] + '\n' }}
 {%- for argument in tool['function']['parameters']['properties'] %}
 {{- argument + ': ' + tool['function']['parameters']['properties'][argument]['description'] + '\n' }}
 {%- endfor %}
 {{- '\n</tool>' }}
 {%- endif %}
{%- endif %}
The specific tokens and tool descriptions your template renders should of course be chosen to match the ones your model
was trained with. There is no requirement that your
model
understands JSON schema input, only that your template can translate
JSON schema into your model’s format. For example,
Command-R
was trained with tools defined using Python function headers, but the Command-R tool template accepts JSON schema,
converts types internally and renders the input tools as Python headers. You can do a lot with templates!
Tool calls
Tool calls, if present, will be a list attached to a message with the “assistant” role. Note that
tool_calls
is
always a list, even though most tool-calling models only support single tool calls at a time, which means
the list will usually only have a single element. Here is a sample message dict containing a tool call:
Copied
{
"role"
:
"assistant"
,
"tool_calls"
:
[
{
"type"
:
"function"
,
"function"
:
{
"name"
:
"multiply"
,
"arguments"
:
{
"a"
:
5
,
"b"
:
6
}
}
}
]
}
And a common pattern for handling them would be something like this:
Copied
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
 {%- for tool_call in message['tool_calls'] %}
 {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
 {%- endif %}
 {%- endfor %}
{%- endif %}
Again, you should render the tool call with the formatting and special tokens that your model expects.
Tool responses
Tool responses have a simple format: They are a message dict with the “tool” role, a “name” key giving the name
of the called function, and a “content” key containing the result of the tool call. Here is a sample tool response:
Copied
{
"role"
:
"tool"
,
"name"
:
"multiply"
,
"content"
:
"30"
}
You don’t need to use all of the keys in the tool response. For example, if your model doesn’t expect the function
name to be included in the tool response, then rendering it can be as simple as:
Copied
{%- if message['role'] == 'tool' %}
 {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
Again, remember that the actual formatting and special tokens are model-specific - you should take a lot of care
to ensure that tokens, whitespace and everything else exactly match the format your model was trained with!
<
>
Update
on GitHub
Transformers
→


---


## Hugging Face Tokenizer Documentation

**Source:** https://huggingface.co/docs/transformers/main_classes/tokenizer

Transformers documentation
Tokenizer
Transformers
🏡 View all docs
AWS Trainium & Inferentia
Accelerate
Argilla
AutoTrain
Bitsandbytes
CLI
Chat UI
Dataset viewer
Datasets
Deploying on AWS
Diffusers
Distilabel
Evaluate
Google Cloud
Google TPUs
Gradio
Hub
Hub Python Library
Huggingface.js
Inference Endpoints (dedicated)
Inference Providers
Kernels
LeRobot
Leaderboards
Lighteval
Microsoft Azure
OpenEnv
Optimum
PEFT
Reachy Mini
Safetensors
Sentence Transformers
TRL
Tasks
Text Embeddings Inference
Text Generation Inference
Tokenizers
Trackio
Transformers
Transformers.js
Xet
smolagents
timm
Search documentation
main
v5.17.0
v5.15.1
v5.14.0
v5.13.1
v5.12.0
v5.11.0
v5.10.4
v5.9.0
v5.8.1
v5.7.0
v5.6.2
v5.5.4
v5.4.0
v5.3.0
v5.2.0
v5.1.0
v5.0.0
v4.57.6
v4.56.2
v4.55.4
v4.53.3
v4.52.3
v4.51.3
v4.50.0
v4.49.0
v4.48.2
v4.47.1
v4.46.3
v4.45.2
v4.44.2
v4.43.4
v4.42.4
v4.41.2
v4.40.2
v4.39.3
v4.38.2
v4.37.2
v4.36.1
v4.35.2
v4.34.1
v4.33.3
v4.32.1
v4.31.0
v4.30.0
v4.29.1
v4.28.1
v4.27.2
v4.26.1
v4.25.1
v4.24.0
v4.23.1
v4.22.2
v4.21.3
v4.20.1
v4.19.4
v4.18.0
v4.17.0
v4.16.2
v4.15.0
v4.14.1
v4.13.0
v4.12.5
v4.11.3
v4.10.1
v4.9.2
v4.8.2
v4.7.0
v4.6.0
v4.5.1
v4.4.2
v4.3.3
v4.2.2
v4.1.1
v4.0.1
v3.5.1
v3.4.0
v3.3.1
v3.2.0
v3.1.0
v3.0.2
v2.11.0
v2.10.0
v2.9.1
v2.8.0
v2.7.0
v2.6.0
v2.5.1
v2.4.1
v2.3.0
v2.2.2
v2.1.1
v2.0.0
v1.2.0
v1.1.0
v1.0.0
doc-builder-html
AR
DE
EN
ES
FR
HI
IT
JA
KO
PT
RO
TR
ZH
Join the Hugging Face community
and get access to the augmented documentation experience
Collaborate on models, datasets and Spaces
Faster examples with accelerated inference
Switch between documentation themes
Sign Up
to get started
Copy page
Tokenizer
A tokenizer is in charge of preparing the inputs for a model. The library contains tokenizers for all the models. Most
of the tokenizers are available in two flavors: a full python implementation and a “Fast” implementation based on the
Rust library
🤗 Tokenizers
. The “Fast” implementations allow:
a significant speed-up in particular when doing batched tokenization and
additional methods to map between the original string (character and words) and the token space (e.g. getting the
index of the token comprising a given character or the span of characters corresponding to a given token).
The base classes
PreTrainedTokenizer
and
PreTrainedTokenizerFast
implement the common methods for encoding string inputs in model inputs (see below) and instantiating/saving python and
“Fast” tokenizers either from a local file or directory or from a pretrained tokenizer provided by the library
(downloaded from HuggingFace’s AWS S3 repository). They both rely on
PreTrainedTokenizerBase
that contains the common methods.
PreTrainedTokenizer
and
PreTrainedTokenizerFast
thus implement the main
methods for using all the tokenizers:
Tokenizing (splitting strings in sub-word token strings), converting tokens strings to ids and back, and
encoding/decoding (i.e., tokenizing and converting to integers).
Adding new tokens to the vocabulary in a way that is independent of the underlying structure (BPE, SentencePiece…).
Managing special tokens (like mask, beginning-of-sentence, etc.): adding them, assigning them to attributes in the
tokenizer for easy access and making sure they are not split during tokenization.
BatchEncoding
holds the output of the
PreTrainedTokenizerBase
’s encoding methods (
__call__
,
encode_plus
and
batch_encode_plus
) and is derived from a Python dictionary. When the tokenizer is a pure python
tokenizer, this class behaves just like a standard python dictionary and holds the various model inputs computed by
these methods (
input_ids
,
attention_mask
…). When the tokenizer is a “Fast” tokenizer (i.e., backed by
HuggingFace
tokenizers library
), this class provides in addition
several advanced alignment methods which can be used to map between the original string (character and words) and the
token space (e.g., getting the index of the token comprising a given character or the span of characters corresponding
to a given token).
Multimodal Tokenizer
Apart from that each tokenizer can be a “multimodal” tokenizer which means that the tokenizer will hold all relevant special tokens
as part of tokenizer attributes for easier access. For example, if the tokenizer is loaded from a vision-language model like LLaVA, you will
be able to access
tokenizer.image_token_id
to obtain the special image token used as a placeholder.
To enable extra special tokens for any type of tokenizer, you have to add the following lines and save the tokenizer. Extra special tokens do not
have to be modality related and can be anything that the model often needs access to. In the below code, tokenizer at
output_dir
will have direct access
to three more special tokens.
Copied
vision_tokenizer = AutoTokenizer.from_pretrained(
"llava-hf/llava-1.5-7b-hf"
,
 extra_special_tokens={
"image_token"
:
"<image>"
,
"boi_token"
:
"<image_start>"
,
"eoi_token"
:
"<image_end>"
}
)
print
(vision_tokenizer.image_token, vision_tokenizer.image_token_id)
(
"<image>"
,
32000
)
PreTrainedTokenizer
class
transformers.
PythonBackend
<
source
>
(
**kwargs
)
Parameters
model_max_length
(
int
,
optional
) —
The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
loaded with
from_pretrained()
, this will be set to the
value stored for the associated model in
max_model_input_sizes
(see above). If no value is provided, will
default to VERY_LARGE_INTEGER (
int(1e30)
).
padding_side
(
str
,
optional
) —
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
truncation_side
(
str
,
optional
) —
The side on which the model should have truncation applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
chat_template
(
str
,
optional
) —
A Jinja template string that will be used to format lists of chat messages. See
https://huggingface.co/docs/transformers/chat_templating
for a full description.
model_input_names
(
list[string]
,
optional
) —
The list of inputs accepted by the forward pass of the model (like
"token_type_ids"
or
"attention_mask"
). Default value is picked from the class attribute of the same name.
bos_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the beginning of a sentence.
eos_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the end of a sentence.
unk_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing an out-of-vocabulary token.
sep_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token separating two different sentences in the same input (used by BERT for instance).
pad_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
attention mechanisms or loss computation.
cls_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the class of the input (used by BERT for instance).
mask_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing a masked token (used by masked-language modeling pretraining objectives, like
BERT). Will be associated to
self.mask_token
and
self.mask_token_id
.
extra_special_tokens
(list of
str
or
tokenizers.AddedToken
,
optional
) —
A list of extra model-specific special tokens. Add them here to ensure they are skipped when decoding with
skip_special_tokens
is set to True. If they are not part of the vocabulary, they will be added at the end
of the vocabulary.
split_special_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not the special tokens should be split during the tokenization process. Passing will affect the
internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
<s>
is the
bos_token
, then
tokenizer.tokenize("<s>") = ['<s>
]. Otherwise, if
split_special_tokens=True
, then
tokenizer.tokenize("<s>")
will be give
['<','s', '>']
.
Base class for all slow tokenizers.
Inherits from
PreTrainedTokenizerBase
.
Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
pretrained tokenizers as well as adding tokens to the vocabulary.
This class also contain the added tokens in a unified way on top of all tokenizers so we don’t have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).
Class attributes (overridden by derived classes)
vocab_files_names
(
dict[str, str]
) — A dictionary with, as keys, the
__init__
keyword name of each
vocabulary file required by the model, and as associated values, the filename for saving the associated file
(string).
pretrained_vocab_files_map
(
dict[str, dict[str, str]]
) — A dictionary of dictionaries, with the
high-level keys being the
__init__
keyword name of each vocabulary file required by the model, the
low-level being the
short-cut-names
of the pretrained models with, as associated values, the
url
to the
associated pretrained vocabulary file.
model_input_names
(
list[str]
) — A list of inputs expected in the forward pass of the model.
padding_side
(
str
) — The default value for the side on which the model should have padding applied.
Should be
'right'
or
'left'
.
truncation_side
(
str
) — The default value for the side on which the model should have truncation
applied. Should be
'right'
or
'left'
.
__call__
<
source
>
(
text
: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None
text_pair
: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None
text_target
: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None
text_pair_target
: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None
add_special_tokens
: bool = True
padding
: bool | str | PaddingStrategy = False
truncation
: bool | str | TruncationStrategy | None = None
max_length
: int | None = None
stride
: int = 0
is_split_into_words
: bool = False
pad_to_multiple_of
: int | None = None
padding_side
: str | None = None
return_tensors
: str | TensorType | None = None
return_token_type_ids
: bool | None = None
return_attention_mask
: bool | None = None
return_overflowing_tokens
: bool = False
return_special_tokens_mask
: bool = False
return_offsets_mapping
: bool = False
return_length
: bool = False
verbose
: bool = True
tokenizer_kwargs
: dict[str, Any] | None = None
**kwargs
)
→
BatchEncoding
Parameters
text
(
str
,
list[str]
,
list[list[str]]
,
optional
) —
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
is_split_into_words=True
(to lift the ambiguity with a batch of sequences).
text_pair
(
str
,
list[str]
,
list[list[str]]
,
optional
) —
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
is_split_into_words=True
(to lift the ambiguity with a batch of sequences).
text_target
(
str
,
list[str]
,
list[list[str]]
,
optional
) —
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set
is_split_into_words=True
(to lift the ambiguity with a batch of sequences).
text_pair_target
(
str
,
list[str]
,
list[list[str]]
,
optional
) —
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set
is_split_into_words=True
(to lift the ambiguity with a batch of sequences).
tokenizer_kwargs
(
dict[str, Any]
,
optional
) —
Additional kwargs to pass to the tokenizer. These will be merged with the explicit parameters and
other kwargs, with explicit parameters taking precedence.
add_special_tokens
(
bool
,
optional
, defaults to
True
) —
Whether or not to add special tokens when encoding the sequences. This will use the underlying
PretrainedTokenizerBase.build_inputs_with_special_tokens
function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add
bos
or
eos
tokens
automatically.
padding
(
bool
,
str
or
PaddingStrategy
,
optional
, defaults to
False
) —
Activates and controls padding. Accepts the following values:
True
or
'longest'
: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).
'max_length'
: Pad to a maximum length specified with the argument
max_length
or to the maximum
acceptable input length for the model if that argument is not provided.
False
or
'do_not_pad'
(default): No padding (i.e., can output a batch with sequences of different
lengths).
truncation
(
bool
,
str
or
TruncationStrategy
,
optional
, defaults to
False
) —
Activates and controls truncation. Accepts the following values:
True
or
'longest_first'
: Truncate to a maximum length specified with the argument
max_length
or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.
'only_first'
: Truncate to a maximum length specified with the argument
max_length
or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
'only_second'
: Truncate to a maximum length specified with the argument
max_length
or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
False
or
'do_not_truncate'
(default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).
max_length
(
int
,
optional
) —
Controls the maximum length to use by one of the truncation/padding parameters.
If left unset or set to
None
, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.
stride
(
int
,
optional
, defaults to 0) —
If set to a number along with
max_length
, the overflowing tokens returned when
return_overflowing_tokens=True
will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.
is_split_into_words
(
bool
,
optional
, defaults to
False
) —
Whether or not the input is already pre-tokenized (e.g., split into words). If set to
True
, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.
pad_to_multiple_of
(
int
,
optional
) —
If set will pad the sequence to a multiple of the provided value. Requires
padding
to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
>= 7.5
(Volta).
padding_side
(
str
,
optional
) —
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
return_tensors
(
str
or
TensorType
,
optional
) —
If set, will return tensors instead of list of python integers. Acceptable values are:
'pt'
: Return PyTorch
torch.Tensor
objects.
'np'
: Return Numpy
np.ndarray
objects.
return_token_type_ids
(
bool
,
optional
) —
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer’s default, defined by the
return_outputs
attribute.
What are token type IDs?
return_attention_mask
(
bool
,
optional
) —
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer’s default, defined by the
return_outputs
attribute.
What are attention masks?
return_overflowing_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with
truncation_strategy = longest_first
or
True
, an error is raised instead
of returning overflowing tokens.
return_special_tokens_mask
(
bool
,
optional
, defaults to
False
) —
Whether or not to return special tokens mask information.
return_offsets_mapping
(
bool
,
optional
, defaults to
False
) —
Whether or not to return
(char_start, char_end)
for each token.
This is only available on fast tokenizers inheriting from
PreTrainedTokenizerFast
, if using
Python’s tokenizer, this method will raise
NotImplementedError
.
return_length
(
bool
,
optional
, defaults to
False
) —
Whether or not to return the lengths of the encoded inputs.
verbose
(
bool
,
optional
, defaults to
True
) —
Whether or not to print more information and warnings.
*
*kwargs
— passed to the
self.tokenize()
method
Returns
BatchEncoding
A
BatchEncoding
with the following fields:
input_ids
— List of token ids to be fed to a model.
What are input IDs?
token_type_ids
— List of token type ids to be fed to a model (when
return_token_type_ids=True
or
if
“token_type_ids”
is in
self.model_input_names
).
What are token type IDs?
attention_mask
— List of indices specifying which tokens should be attended to by the model (when
return_attention_mask=True
or if
“attention_mask”
is in
self.model_input_names
).
What are attention masks?
overflowing_tokens
— List of overflowing tokens sequences (when a
max_length
is specified and
return_overflowing_tokens=True
).
num_truncated_tokens
— Number of tokens truncated (when a
max_length
is specified and
return_overflowing_tokens=True
).
special_tokens_mask
— List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
regular sequence tokens (when
add_special_tokens=True
and
return_special_tokens_mask=True
).
length
— The length of the inputs (when
return_length=True
)
Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.
add_tokens
<
source
>
(
new_tokens
: str | AddedToken | Sequence[str | AddedToken]
special_tokens
: bool = False
)
→
int
Parameters
new_tokens
(
str
,
tokenizers.AddedToken
or a sequence of
str
or
tokenizers.AddedToken
) —
Tokens are only added if they are not already in the vocabulary.
tokenizers.AddedToken
wraps a string
token to let you personalize its behavior: whether this token should only match against a single word,
whether this token should strip all potential whitespaces on the left side, whether this token should
strip all potential whitespaces on the right side, etc.
special_tokens
(
bool
,
optional
, defaults to
False
) —
Specifies if the token is special. This mostly changes the normalization behavior
See details for
tokenizers.AddedToken
in HuggingFace tokenizers library.
Returns
int
Number of tokens added to the vocabulary.
#TODO remove this from here! PreTrainedTokenizerBase should be agnostic of AddedToken.
Add a list of new tokens. If the new tokens are not in the vocabulary, they are added to the end. Added tokens and
tokens from the vocabulary of the tokenization algorithm are therefore not treated in the same way.
Examples:
Copied
# Let's see how to increase the vocabulary of Bert model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained(
"google-bert/bert-base-uncased"
)
model = BertModel.from_pretrained(
"google-bert/bert-base-uncased"
)

num_added_toks = tokenizer.add_tokens([
"new_tok1"
,
"my_new-tok2"
])
print
(
"We have added"
, num_added_toks,
"tokens"
)
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(
len
(tokenizer))
add_special_tokens
<
source
>
(
special_tokens_dict
: dict[str, str | AddedToken | Sequence[str | AddedToken]]
replace_extra_special_tokens
= True
)
→
int
Parameters
special_tokens_dict
(dictionary
str
to
str
,
tokenizers.AddedToken
, or
Sequence[Union[str, AddedToken]]
) —
Keys should be in the list of predefined special attributes: [
bos_token
,
eos_token
,
unk_token
,
sep_token
,
pad_token
,
cls_token
,
mask_token
,
extra_special_tokens
].
Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
assign the index of the
unk_token
to them).
replace_extra_special_tokens
(
bool
,
optional
, defaults to
True
) —
If
True
, the existing list of extra special tokens will be replaced by the list provided in
special_tokens_dict
. Otherwise,
extra_special_tokens
will be extended. In the former
case, the tokens will NOT be removed from the tokenizer’s full vocabulary - they are only being flagged
as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
added_tokens_encoder
and
added_tokens_decoder
. This means that the previous
extra_special_tokens
are still added tokens, and will not be split by the model.
Returns
int
Number of tokens added to the vocabulary.
Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
current vocabulary).
When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
model so that its embedding matrix matches the tokenizer.
In order to do that, please use the
resize_token_embeddings()
method.
Using
add_special_tokens
will ensure your special tokens can be used in several ways:
Special tokens can be skipped when decoding using
skip_special_tokens = True
.
Special tokens are carefully handled by the tokenizer (they are never split), similar to
AddedTokens
.
You can easily refer to special tokens using tokenizer class attributes like
tokenizer.cls_token
. This
makes it easy to develop model-agnostic training and fine-tuning scripts.
When possible, special tokens are already registered for provided pretrained models (for instance
BertTokenizer
cls_token
is already registered to be
'[CLS]'
and XLM’s one is also registered to be
'</s>'
).
Examples:
Copied
# Let's see how to add a new classification token to GPT-2
tokenizer = GPT2Tokenizer.from_pretrained(
"openai-community/gpt2"
)
model = GPT2Model.from_pretrained(
"openai-community/gpt2"
)

special_tokens_dict = {
"cls_token"
:
"<CLS>"
}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print
(
"We have added"
, num_added_toks,
"tokens"
)
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(
len
(tokenizer))
assert
tokenizer.cls_token ==
"<CLS>"
apply_chat_template
<
source
>
(
conversation
: list[dict[str, str]] | list[list[dict[str, str]]]
tools
: list[dict | Callable] | None = None
documents
: list[dict[str, str]] | None = None
chat_template
: str | None = None
add_generation_prompt
: bool = False
continue_final_message
: bool | str = False
tokenize
: bool = True
padding
: bool | str | PaddingStrategy = False
truncation
: bool = False
max_length
: int | None = None
return_tensors
: str | TensorType | None = None
return_dict
: bool = True
return_assistant_tokens_mask
: bool = False
tokenizer_kwargs
: dict[str, Any] | None = None
**kwargs
)
→
Union[list[int], Dict]
Parameters
conversation
(Union[list[dict[str, str]], list[list[dict[str, str]]]]) — A list of dicts
with “role” and “content” keys, representing the chat history so far.
tools
(
list[Union[Dict, Callable]]
,
optional
) —
A list of tools (callable functions) that will be accessible to the model. If the template does not
support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
giving the name, description and argument types for the tool. See our
tool use guide
for more information.
documents
(
list[dict[str, str]]
,
optional
) —
A list of dicts representing documents that will be accessible to the model if it is performing RAG
(retrieval-augmented generation). If the template does not support RAG, this argument will have no
effect. We recommend that each document should be a dict containing “title” and “text” keys.
chat_template
(
str
,
optional
) —
A Jinja template to use for this conversion. It is usually not necessary to pass anything to this
argument, as the model’s template will be used by default.
add_generation_prompt
(bool,
optional
) —
If this is set, a prompt with the token(s) that indicate
the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
Note that this argument will be passed to the chat template, and so it must be supported in the
template for this argument to have any effect.
continue_final_message
(bool or str,
optional
) —
If this is set, the chat will be formatted so that the final
message in the chat is open-ended, without any EOS tokens. The model will continue this message
rather than starting a new one. This allows you to “prefill” part of
the model’s response for it. If a string is passed, it will be used as the key for the field to continue
(e.g. “reasoning_content”). Cannot be used at the same time as
add_generation_prompt
.
tokenize
(
bool
, defaults to
True
) —
Whether to tokenize the output. If
False
, the output will be a string.
padding
(
bool
,
str
or
PaddingStrategy
,
optional
, defaults to
False
) —
Select a strategy to pad the returned sequences (according to the model’s padding side and padding
index) among:
True
or
'longest'
: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).
'max_length'
: Pad to a maximum length specified with the argument
max_length
or to the maximum
acceptable input length for the model if that argument is not provided.
False
or
'do_not_pad'
(default): No padding (i.e., can output a batch with sequences of different
lengths).
truncation
(
bool
, defaults to
False
) —
Whether to truncate sequences at the maximum length. Has no effect if tokenize is
False
.
max_length
(
int
,
optional
) —
Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is
False
. If
not specified, the tokenizer’s
max_length
attribute will be used as a default.
return_tensors
(
str
or
TensorType
,
optional
) —
If set, will return tensors of a particular framework. Has no effect if tokenize is
False
. Acceptable
values are:
'pt'
: Return PyTorch
torch.Tensor
objects.
'np'
: Return NumPy
np.ndarray
objects.
return_dict
(
bool
, defaults to
True
) —
Whether to return a dictionary with named outputs. Has no effect if tokenize is
False
.
tokenizer_kwargs
(
dict[str -- Any]
,
optional
): Additional kwargs to pass to the tokenizer.
return_assistant_tokens_mask
(
bool
, defaults to
False
) —
Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
the mask will contain 1. For user and system tokens, the mask will contain 0.
This functionality is only available for chat templates that support it via the
{% generation %}
keyword.
*
*kwargs
— Additional kwargs to pass to the template renderer. Will be accessible by the chat template.
Returns
Union[list[int], Dict]
A list of token ids representing the tokenized chat so far, including control tokens. This
output is ready to pass to the model, either directly or via methods like
generate()
. If
return_dict
is
set, will return a dict of tokenizer outputs instead.
Converts a list of dictionaries with
"role"
and
"content"
keys to a list of token
ids. This method is intended for use with chat models, and will read the tokenizer’s chat_template attribute to
determine the format and control tokens to use when converting.
batch_decode
<
source
>
(
sequences
: list[int] | list[list[int]] | np.ndarray | torch.Tensor
skip_special_tokens
: bool = False
clean_up_tokenization_spaces
: bool | None = None
**kwargs
)
→
list[str]
Parameters
sequences
(
Union[list[int], list[list[int]], np.ndarray, torch.Tensor]
) —
List of tokenized input ids. Can be obtained using the
__call__
method.
skip_special_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not to remove special tokens in the decoding.
clean_up_tokenization_spaces
(
bool
,
optional
) —
Whether or not to clean up the tokenization spaces. If
None
, will default to
self.clean_up_tokenization_spaces
.
kwargs
(additional keyword arguments,
optional
) —
Will be passed to the underlying model specific decode method.
Returns
list[str]
The list of decoded sentences.
Convert a list of lists of token ids into a list of strings by calling decode.
This method is provided for backwards compatibility. The
decode
method now handles batched input natively,
so you can use
decode
directly instead of
batch_decode
.
decode
<
source
>
(
token_ids
: int | list[int] | list[list[int]] | np.ndarray | torch.Tensor
skip_special_tokens
: bool = False
**kwargs
)
→
Union[str, list[str]]
Parameters
token_ids
(
Union[int, list[int], list[list[int]], np.ndarray, torch.Tensor]
) —
A single sequence or a batch (list of sequences) of tokenized input ids. Can be obtained using the
__call__
method.
skip_special_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not to remove special tokens in the decoding.
kwargs
(additional keyword arguments,
optional
) —
Will be passed to the underlying model specific decode method.
Returns
Union[str, list[str]]
The decoded string for a single sequence, or a list of decoded strings for a
batch of sequences.
Converts a sequence of ids into a string, or a list of sequences into a list of strings,
using the tokenizer and vocabulary with options to remove special tokens and clean up
tokenization spaces.
Similar to doing
self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))
.
encode
<
source
>
(
text
: TextInput | PreTokenizedInput | EncodedInput
text_pair
: TextInput | PreTokenizedInput | EncodedInput | None = None
add_special_tokens
: bool = True
padding
: bool | str | PaddingStrategy = False
truncation
: bool | str | TruncationStrategy | None = None
max_length
: int | None = None
stride
: int = 0
padding_side
: str | None = None
return_tensors
: str | TensorType | None = None
**kwargs
)
→
list[int]
,
torch.Tensor
, or
np.ndarray
Parameters
text
(
str
,
list[str]
or
list[int]
) —
The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
tokenize
method) or a list of integers (tokenized string ids using the
convert_tokens_to_ids
method).
text_pair
(
str
,
list[str]
or
list[int]
,
optional
) —
Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
the
tokenize
method) or a list of integers (tokenized string ids using the
convert_tokens_to_ids
method).
add_special_tokens
(
bool
,
optional
, defaults to
True
) —
Whether or not to add special tokens when encoding the sequences. This will use the underlying
PretrainedTokenizerBase.build_inputs_with_special_tokens
function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add
bos
or
eos
tokens
automatically.
padding
(
bool
,
str
or
PaddingStrategy
,
optional
, defaults to
False
) —
Activates and controls padding. Accepts the following values:
True
or
'longest'
: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).
'max_length'
: Pad to a maximum length specified with the argument
max_length
or to the maximum
acceptable input length for the model if that argument is not provided.
False
or
'do_not_pad'
(default): No padding (i.e., can output a batch with sequences of different
lengths).
truncation
(
bool
,
str
or
TruncationStrategy
,
optional
, defaults to
False
) —
Activates and controls truncation. Accepts the following values:
True
or
'longest_first'
: Truncate to a maximum length specified with the argument
max_length
or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.
'only_first'
: Truncate to a maximum length specified with the argument
max_length
or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
'only_second'
: Truncate to a maximum length specified with the argument
max_length
or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
False
or
'do_not_truncate'
(default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).
max_length
(
int
,
optional
) —
Controls the maximum length to use by one of the truncation/padding parameters.
If left unset or set to
None
, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.
stride
(
int
,
optional
, defaults to 0) —
If set to a number along with
max_length
, the overflowing tokens returned when
return_overflowing_tokens=True
will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.
is_split_into_words
(
bool
,
optional
, defaults to
False
) —
Whether or not the input is already pre-tokenized (e.g., split into words). If set to
True
, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.
pad_to_multiple_of
(
int
,
optional
) —
If set will pad the sequence to a multiple of the provided value. Requires
padding
to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
>= 7.5
(Volta).
padding_side
(
str
,
optional
) —
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
return_tensors
(
str
or
TensorType
,
optional
) —
If set, will return tensors instead of list of python integers. Acceptable values are:
'pt'
: Return PyTorch
torch.Tensor
objects.
'np'
: Return Numpy
np.ndarray
objects.
*
*kwargs
— Passed along to the
.tokenize()
method.
Returns
list[int]
,
torch.Tensor
, or
np.ndarray
The tokenized ids of the text.
Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.
Same as doing
self.convert_tokens_to_ids(self.tokenize(text))
.
push_to_hub
<
source
>
(
repo_id
: str
commit_message
: str | None = None
commit_description
: str | None = None
private
: bool | None = None
token
: bool | str | None = None
revision
: str | None = None
create_pr
: bool = False
max_shard_size
: int | str | None = '50GB'
tags
: list[str] | None = None
)
Parameters
repo_id
(
str
) —
The name of the repository you want to push your tokenizer to. It should contain your organization name
when pushing to a given organization.
commit_message
(
str
,
optional
) —
Message to commit while pushing. Will default to
"Upload tokenizer"
.
commit_description
(
str
,
optional
) —
The description of the commit that will be created
private
(
bool
,
optional
) —
Whether to make the repo private. If
None
(default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
token
(
bool
or
str
,
optional
) —
The token to use as HTTP bearer authorization for remote files. If
True
(default), will use the token generated
when running
hf auth login
(stored in
~/.huggingface
).
revision
(
str
,
optional
) —
Branch to push the uploaded files to.
create_pr
(
bool
,
optional
, defaults to
False
) —
Whether or not to create a PR with the uploaded files or directly commit.
max_shard_size
(
int
or
str
,
optional
, defaults to
"50GB"
) —
Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
will then be each of size lower than this size. If expressed as a string, needs to be digits followed
by a unit (like
"5MB"
).
tags
(
list[str]
,
optional
) —
List of tags to push on the Hub.
Upload the tokenizer files to the 🤗 Model Hub.
Examples:
Copied
from
transformers
import
AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
"google-bert/bert-base-cased"
)
# Push the tokenizer to your namespace with the name "my-finetuned-bert".
tokenizer.push_to_hub(
"my-finetuned-bert"
)
# Push the tokenizer to an organization with the name "my-finetuned-bert".
tokenizer.push_to_hub(
"huggingface/my-finetuned-bert"
)
build_inputs_with_special_tokens
<
source
>
(
token_ids_0
: list
token_ids_1
: list[int] | None = None
)
→
list[int]
Parameters
token_ids_0
(
list[int]
) —
List of IDs to which the special tokens will be added.
token_ids_1
(
list[int]
,
optional
) —
Optional second list of IDs for sequence pairs.
Returns
list[int]
List of input IDs with the appropriate special tokens.
Build model inputs from a sequence or a pair of sequences by adding special tokens.
This method dynamically builds inputs based on the tokenizer’s
special_tokens_pattern
:
"none"
: No special tokens
"cls_sep"
: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
"eos"
: seq0 [EOS] or seq0 [EOS] seq1 [EOS]
"bos"
: [BOS] seq0 or [BOS] seq0 [BOS] seq1
"bos_eos"
: [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
"cls_double_sep"
: [CLS] seq0 [SEP] or [CLS] seq0 [SEP][SEP] seq1 [SEP]
"prefix_suffix"
:
<prefix_tokens> seq0 [seq1] <suffix_tokens>
(custom prefix/suffix stored on the tokenizer)
create_token_type_ids_from_sequences
<
source
>
(
token_ids_0
: list
token_ids_1
: list[int] | None = None
)
→
list[int]
Parameters
token_ids_0
(
list[int]
) —
List of IDs.
token_ids_1
(
list[int]
,
optional
) —
Optional second list of IDs for sequence pairs.
Returns
list[int]
Token type IDs according to the configured pattern.
Create a mask from the two sequences passed to be used in a sequence-pair classification task.
This method dynamically builds the token type IDs based on the tokenizer’s configuration attributes:
token_type_ids_pattern
: Pattern to use (“all_zeros” or “bert_style”)
token_type_ids_include_special_tokens
: Whether to account for special tokens in length calculation
Examples:
Copied
# All zeros pattern (default, used by RoBERTa, BART, etc.)
tokenizer.token_type_ids_pattern =
"all_zeros"
# Returns: [0, 0, 0, ...] for both sequences
# BERT-style pattern (first sequence gets 0s, second gets 1s)
tokenizer.token_type_ids_pattern =
"bert_style"
# Returns: [0, 0, 0, ..., 1, 1, 1, ...] for sequence pairs
get_added_vocab
<
source
>
(
)
→
dict[str, int]
Returns
dict[str, int]
The added tokens.
Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
something we should change.
get_special_tokens_mask
<
source
>
(
token_ids_0
: list
token_ids_1
: list | None = None
already_has_special_tokens
: bool = False
)
→
A list of integers in the range [0, 1]
Parameters
token_ids_0
(
list[int]
) —
List of ids of the first sequence.
token_ids_1
(
list[int]
,
optional
) —
List of ids of the second sequence.
already_has_special_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not the token list is already formatted with special tokens for the model.
Returns
A list of integers in the range [0, 1]
1 for a special token, 0 for a sequence token.
Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer
prepare_for_model
or
encode_plus
methods.
This method dynamically builds the special tokens mask based on the tokenizer’s
special_tokens_pattern
:
"none"
: No special tokens (default, returns all 0s)
"cls_sep"
: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
"eos"
: seq0 [EOS] or seq0 [EOS] seq1 [EOS]
"bos"
: [BOS] seq0 or [BOS] seq0 [BOS] seq1
"bos_eos"
: [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
"cls_double_sep"
: [CLS] seq0 [SEP] or [CLS] seq0 [SEP][SEP] seq1 [SEP]
"prefix_suffix"
:
<prefix_tokens> seq0 [seq1] <suffix_tokens>
num_special_tokens_to_add
<
source
>
(
pair
: bool = False
)
→
int
Parameters
pair
(
bool
,
optional
, defaults to
False
) —
Whether the number of added tokens should be computed in the case of a sequence pair or a single
sequence.
Returns
int
Number of special tokens added to sequences.
Returns the number of added tokens when encoding a sequence with special tokens.
This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
this inside your training loop.
prepare_for_model
<
source
>
(
ids
: list
pair_ids
: list[int] | None = None
add_special_tokens
: bool = True
padding
: bool | str | transformers.utils.generic.PaddingStrategy = False
truncation
: bool | str | transformers.tokenization_utils_base.TruncationStrategy = False
max_length
: int | None = None
stride
: int = 0
pad_to_multiple_of
: int | None = None
padding_side
: str | None = None
return_tensors
: str | transformers.utils.generic.TensorType | None = None
return_token_type_ids
: bool | None = None
return_attention_mask
: bool | None = None
return_overflowing_tokens
: bool = False
return_special_tokens_mask
: bool = False
return_length
: bool = False
verbose
: bool = True
prepend_batch_axis
: bool = False
**kwargs
)
Parameters
ids
— Tokenized input ids of the first sequence.
pair_ids
— Tokenized input ids of the second sequence (optional).
Prepares a sequence of input ids so it can be used by the model. Adds special tokens, truncates, and pads.
prepare_for_tokenization
<
source
>
(
text
: str
is_split_into_words
: bool = False
**kwargs
)
→
tuple[str, dict[str, Any]]
Parameters
text
(
str
) —
The text to prepare.
is_split_into_words
(
bool
,
optional
, defaults to
False
) —
Whether or not the input is already pre-tokenized (e.g., split into words). If set to
True
, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.
kwargs
(
dict[str, Any]
,
optional
) —
Keyword arguments to use for the tokenization.
Returns
tuple[str, dict[str, Any]]
The prepared text and the unused kwargs.
Performs any necessary transformations before tokenization.
This method should pop the arguments from kwargs and return the remaining
kwargs
as well. We test the
kwargs
at the end of the encoding process to be sure all the arguments have been used.
save_vocabulary
<
source
>
(
save_directory
: str
filename_prefix
: str | None = None
)
→
tuple[str, ...]
Parameters
save_directory
(
str
) —
The directory in which to save the vocabulary.
filename_prefix
(
str
,
optional
) —
An optional prefix to add to the named of the saved files.
Returns
tuple[str, ...]
Paths to the files saved, or empty tuple if no files saved.
Default implementation for common vocabulary saving patterns.
Saves self.encoder/self.vocab as JSON, optionally with self.bpe_ranks as merges.
Returns empty tuple if no vocabulary exists.
Override this method if your tokenizer needs custom saving logic (e.g., SentencePiece models,
multiple vocabulary files, or special file formats).
tokenize
<
source
>
(
text
: str
**kwargs
)
Parameters
text
— The sequence to be encoded.
*
*kwargs
— Passed along to the model-specific
prepare_for_tokenization
preprocessing method.
Converts a string into a sequence of tokens, using the tokenizer.
truncate_sequences
<
source
>
(
ids
: list
pair_ids
: list[int] | None = None
num_tokens_to_remove
: int = 0
truncation_strategy
: str | transformers.tokenization_utils_base.TruncationStrategy = 'longest_first'
stride
: int = 0
)
Truncates sequences according to the specified strategy.
PreTrainedTokenizerFast
The
PreTrainedTokenizerFast
depends on the
tokenizers
library. The tokenizers obtained from the 🤗 tokenizers library can be
loaded very simply into 🤗 transformers. Take a look at the
Using tokenizers from 🤗 tokenizers
page to understand how this is done.
class
transformers.
TokenizersBackend
<
source
>
(
*args
**kwargs
)
Parameters
model_max_length
(
int
,
optional
) —
The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
loaded with
from_pretrained()
, this will be set to the
value stored for the associated model in
max_model_input_sizes
(see above). If no value is provided, will
default to VERY_LARGE_INTEGER (
int(1e30)
).
padding_side
(
str
,
optional
) —
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
truncation_side
(
str
,
optional
) —
The side on which the model should have truncation applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
chat_template
(
str
,
optional
) —
A Jinja template string that will be used to format lists of chat messages. See
https://huggingface.co/docs/transformers/chat_templating
for a full description.
model_input_names
(
list[string]
,
optional
) —
The list of inputs accepted by the forward pass of the model (like
"token_type_ids"
or
"attention_mask"
). Default value is picked from the class attribute of the same name.
bos_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the beginning of a sentence.
eos_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the end of a sentence.
unk_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing an out-of-vocabulary token.
sep_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token separating two different sentences in the same input (used by BERT for instance).
pad_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
attention mechanisms or loss computation.
cls_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the class of the input (used by BERT for instance).
mask_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing a masked token (used by masked-language modeling pretraining objectives, like
BERT). Will be associated to
self.mask_token
and
self.mask_token_id
.
extra_special_tokens
(list of
str
or
tokenizers.AddedToken
,
optional
) —
A list of extra model-specific special tokens. Add them here to ensure they are skipped when decoding with
skip_special_tokens
is set to True. If they are not part of the vocabulary, they will be added at the end
of the vocabulary.
split_special_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not the special tokens should be split during the tokenization process. Passing will affect the
internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
<s>
is the
bos_token
, then
tokenizer.tokenize("<s>") = ['<s>
]. Otherwise, if
split_special_tokens=True
, then
tokenizer.tokenize("<s>")
will be give
['<','s', '>']
.
tokenizer_object
(
tokenizers.Tokenizer
) —
A
tokenizers.Tokenizer
object from 🤗 tokenizers to instantiate from. See
Using tokenizers from 🤗
tokenizers
for more information.
tokenizer_file
(
str
) —
A path to a local JSON file representing a previously serialized
tokenizers.Tokenizer
object from 🤗
tokenizers.
Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).
Inherits from
PreTrainedTokenizerBase
.
Handles all the shared methods for tokenization and special tokens, as well as methods for
downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.
This class also contains the added tokens in a unified way on top of all tokenizers so we don’t have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).
Class attributes (overridden by derived classes)
vocab_files_names
(
dict[str, str]
) — A dictionary with, as keys, the
__init__
keyword name of each
vocabulary file required by the model, and as associated values, the filename for saving the associated file
(string).
pretrained_vocab_files_map
(
dict[str, dict[str, str]]
) — A dictionary of dictionaries, with the
high-level keys being the
__init__
keyword name of each vocabulary file required by the model, the
low-level being the
short-cut-names
of the pretrained models with, as associated values, the
url
to the
associated pretrained vocabulary file.
model_input_names
(
list[str]
) — A list of inputs expected in the forward pass of the model.
padding_side
(
str
) — The default value for the side on which the model should have padding applied.
Should be
'right'
or
'left'
.
truncation_side
(
str
) — The default value for the side on which the model should have truncation
applied. Should be
'right'
or
'left'
.
__call__
<
source
>
(
text
: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None
text_pair
: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None
text_target
: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None
text_pair_target
: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None
add_special_tokens
: bool = True
padding
: bool | str | PaddingStrategy = False
truncation
: bool | str | TruncationStrategy | None = None
max_length
: int | None = None
stride
: int = 0
is_split_into_words
: bool = False
pad_to_multiple_of
: int | None = None
padding_side
: str | None = None
return_tensors
: str | TensorType | None = None
return_token_type_ids
: bool | None = None
return_attention_mask
: bool | None = None
return_overflowing_tokens
: bool = False
return_special_tokens_mask
: bool = False
return_offsets_mapping
: bool = False
return_length
: bool = False
verbose
: bool = True
tokenizer_kwargs
: dict[str, Any] | None = None
**kwargs
)
→
BatchEncoding
Parameters
text
(
str
,
list[str]
,
list[list[str]]
,
optional
) —
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
is_split_into_words=True
(to lift the ambiguity with a batch of sequences).
text_pair
(
str
,
list[str]
,
list[list[str]]
,
optional
) —
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
is_split_into_words=True
(to lift the ambiguity with a batch of sequences).
text_target
(
str
,
list[str]
,
list[list[str]]
,
optional
) —
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set
is_split_into_words=True
(to lift the ambiguity with a batch of sequences).
text_pair_target
(
str
,
list[str]
,
list[list[str]]
,
optional
) —
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set
is_split_into_words=True
(to lift the ambiguity with a batch of sequences).
tokenizer_kwargs
(
dict[str, Any]
,
optional
) —
Additional kwargs to pass to the tokenizer. These will be merged with the explicit parameters and
other kwargs, with explicit parameters taking precedence.
add_special_tokens
(
bool
,
optional
, defaults to
True
) —
Whether or not to add special tokens when encoding the sequences. This will use the underlying
PretrainedTokenizerBase.build_inputs_with_special_tokens
function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add
bos
or
eos
tokens
automatically.
padding
(
bool
,
str
or
PaddingStrategy
,
optional
, defaults to
False
) —
Activates and controls padding. Accepts the following values:
True
or
'longest'
: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).
'max_length'
: Pad to a maximum length specified with the argument
max_length
or to the maximum
acceptable input length for the model if that argument is not provided.
False
or
'do_not_pad'
(default): No padding (i.e., can output a batch with sequences of different
lengths).
truncation
(
bool
,
str
or
TruncationStrategy
,
optional
, defaults to
False
) —
Activates and controls truncation. Accepts the following values:
True
or
'longest_first'
: Truncate to a maximum length specified with the argument
max_length
or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.
'only_first'
: Truncate to a maximum length specified with the argument
max_length
or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
'only_second'
: Truncate to a maximum length specified with the argument
max_length
or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
False
or
'do_not_truncate'
(default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).
max_length
(
int
,
optional
) —
Controls the maximum length to use by one of the truncation/padding parameters.
If left unset or set to
None
, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.
stride
(
int
,
optional
, defaults to 0) —
If set to a number along with
max_length
, the overflowing tokens returned when
return_overflowing_tokens=True
will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.
is_split_into_words
(
bool
,
optional
, defaults to
False
) —
Whether or not the input is already pre-tokenized (e.g., split into words). If set to
True
, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.
pad_to_multiple_of
(
int
,
optional
) —
If set will pad the sequence to a multiple of the provided value. Requires
padding
to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
>= 7.5
(Volta).
padding_side
(
str
,
optional
) —
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
return_tensors
(
str
or
TensorType
,
optional
) —
If set, will return tensors instead of list of python integers. Acceptable values are:
'pt'
: Return PyTorch
torch.Tensor
objects.
'np'
: Return Numpy
np.ndarray
objects.
return_token_type_ids
(
bool
,
optional
) —
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer’s default, defined by the
return_outputs
attribute.
What are token type IDs?
return_attention_mask
(
bool
,
optional
) —
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer’s default, defined by the
return_outputs
attribute.
What are attention masks?
return_overflowing_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with
truncation_strategy = longest_first
or
True
, an error is raised instead
of returning overflowing tokens.
return_special_tokens_mask
(
bool
,
optional
, defaults to
False
) —
Whether or not to return special tokens mask information.
return_offsets_mapping
(
bool
,
optional
, defaults to
False
) —
Whether or not to return
(char_start, char_end)
for each token.
This is only available on fast tokenizers inheriting from
PreTrainedTokenizerFast
, if using
Python’s tokenizer, this method will raise
NotImplementedError
.
return_length
(
bool
,
optional
, defaults to
False
) —
Whether or not to return the lengths of the encoded inputs.
verbose
(
bool
,
optional
, defaults to
True
) —
Whether or not to print more information and warnings.
*
*kwargs
— passed to the
self.tokenize()
method
Returns
BatchEncoding
A
BatchEncoding
with the following fields:
input_ids
— List of token ids to be fed to a model.
What are input IDs?
token_type_ids
— List of token type ids to be fed to a model (when
return_token_type_ids=True
or
if
“token_type_ids”
is in
self.model_input_names
).
What are token type IDs?
attention_mask
— List of indices specifying which tokens should be attended to by the model (when
return_attention_mask=True
or if
“attention_mask”
is in
self.model_input_names
).
What are attention masks?
overflowing_tokens
— List of overflowing tokens sequences (when a
max_length
is specified and
return_overflowing_tokens=True
).
num_truncated_tokens
— Number of tokens truncated (when a
max_length
is specified and
return_overflowing_tokens=True
).
special_tokens_mask
— List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
regular sequence tokens (when
add_special_tokens=True
and
return_special_tokens_mask=True
).
length
— The length of the inputs (when
return_length=True
)
Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.
add_tokens
<
source
>
(
new_tokens
: str | AddedToken | Sequence[str | AddedToken]
special_tokens
: bool = False
)
→
int
Parameters
new_tokens
(
str
,
tokenizers.AddedToken
or a sequence of
str
or
tokenizers.AddedToken
) —
Tokens are only added if they are not already in the vocabulary.
tokenizers.AddedToken
wraps a string
token to let you personalize its behavior: whether this token should only match against a single word,
whether this token should strip all potential whitespaces on the left side, whether this token should
strip all potential whitespaces on the right side, etc.
special_tokens
(
bool
,
optional
, defaults to
False
) —
Specifies if the token is special. This mostly changes the normalization behavior
See details for
tokenizers.AddedToken
in HuggingFace tokenizers library.
Returns
int
Number of tokens added to the vocabulary.
#TODO remove this from here! PreTrainedTokenizerBase should be agnostic of AddedToken.
Add a list of new tokens. If the new tokens are not in the vocabulary, they are added to the end. Added tokens and
tokens from the vocabulary of the tokenization algorithm are therefore not treated in the same way.
Examples:
Copied
# Let's see how to increase the vocabulary of Bert model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained(
"google-bert/bert-base-uncased"
)
model = BertModel.from_pretrained(
"google-bert/bert-base-uncased"
)

num_added_toks = tokenizer.add_tokens([
"new_tok1"
,
"my_new-tok2"
])
print
(
"We have added"
, num_added_toks,
"tokens"
)
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(
len
(tokenizer))
add_special_tokens
<
source
>
(
special_tokens_dict
: dict[str, str | AddedToken | Sequence[str | AddedToken]]
replace_extra_special_tokens
= True
)
→
int
Parameters
special_tokens_dict
(dictionary
str
to
str
,
tokenizers.AddedToken
, or
Sequence[Union[str, AddedToken]]
) —
Keys should be in the list of predefined special attributes: [
bos_token
,
eos_token
,
unk_token
,
sep_token
,
pad_token
,
cls_token
,
mask_token
,
extra_special_tokens
].
Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
assign the index of the
unk_token
to them).
replace_extra_special_tokens
(
bool
,
optional
, defaults to
True
) —
If
True
, the existing list of extra special tokens will be replaced by the list provided in
special_tokens_dict
. Otherwise,
extra_special_tokens
will be extended. In the former
case, the tokens will NOT be removed from the tokenizer’s full vocabulary - they are only being flagged
as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
added_tokens_encoder
and
added_tokens_decoder
. This means that the previous
extra_special_tokens
are still added tokens, and will not be split by the model.
Returns
int
Number of tokens added to the vocabulary.
Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
current vocabulary).
When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
model so that its embedding matrix matches the tokenizer.
In order to do that, please use the
resize_token_embeddings()
method.
Using
add_special_tokens
will ensure your special tokens can be used in several ways:
Special tokens can be skipped when decoding using
skip_special_tokens = True
.
Special tokens are carefully handled by the tokenizer (they are never split), similar to
AddedTokens
.
You can easily refer to special tokens using tokenizer class attributes like
tokenizer.cls_token
. This
makes it easy to develop model-agnostic training and fine-tuning scripts.
When possible, special tokens are already registered for provided pretrained models (for instance
BertTokenizer
cls_token
is already registered to be
'[CLS]'
and XLM’s one is also registered to be
'</s>'
).
Examples:
Copied
# Let's see how to add a new classification token to GPT-2
tokenizer = GPT2Tokenizer.from_pretrained(
"openai-community/gpt2"
)
model = GPT2Model.from_pretrained(
"openai-community/gpt2"
)

special_tokens_dict = {
"cls_token"
:
"<CLS>"
}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print
(
"We have added"
, num_added_toks,
"tokens"
)
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(
len
(tokenizer))
assert
tokenizer.cls_token ==
"<CLS>"
apply_chat_template
<
source
>
(
conversation
: list[dict[str, str]] | list[list[dict[str, str]]]
tools
: list[dict | Callable] | None = None
documents
: list[dict[str, str]] | None = None
chat_template
: str | None = None
add_generation_prompt
: bool = False
continue_final_message
: bool | str = False
tokenize
: bool = True
padding
: bool | str | PaddingStrategy = False
truncation
: bool = False
max_length
: int | None = None
return_tensors
: str | TensorType | None = None
return_dict
: bool = True
return_assistant_tokens_mask
: bool = False
tokenizer_kwargs
: dict[str, Any] | None = None
**kwargs
)
→
Union[list[int], Dict]
Parameters
conversation
(Union[list[dict[str, str]], list[list[dict[str, str]]]]) — A list of dicts
with “role” and “content” keys, representing the chat history so far.
tools
(
list[Union[Dict, Callable]]
,
optional
) —
A list of tools (callable functions) that will be accessible to the model. If the template does not
support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
giving the name, description and argument types for the tool. See our
tool use guide
for more information.
documents
(
list[dict[str, str]]
,
optional
) —
A list of dicts representing documents that will be accessible to the model if it is performing RAG
(retrieval-augmented generation). If the template does not support RAG, this argument will have no
effect. We recommend that each document should be a dict containing “title” and “text” keys.
chat_template
(
str
,
optional
) —
A Jinja template to use for this conversion. It is usually not necessary to pass anything to this
argument, as the model’s template will be used by default.
add_generation_prompt
(bool,
optional
) —
If this is set, a prompt with the token(s) that indicate
the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
Note that this argument will be passed to the chat template, and so it must be supported in the
template for this argument to have any effect.
continue_final_message
(bool or str,
optional
) —
If this is set, the chat will be formatted so that the final
message in the chat is open-ended, without any EOS tokens. The model will continue this message
rather than starting a new one. This allows you to “prefill” part of
the model’s response for it. If a string is passed, it will be used as the key for the field to continue
(e.g. “reasoning_content”). Cannot be used at the same time as
add_generation_prompt
.
tokenize
(
bool
, defaults to
True
) —
Whether to tokenize the output. If
False
, the output will be a string.
padding
(
bool
,
str
or
PaddingStrategy
,
optional
, defaults to
False
) —
Select a strategy to pad the returned sequences (according to the model’s padding side and padding
index) among:
True
or
'longest'
: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).
'max_length'
: Pad to a maximum length specified with the argument
max_length
or to the maximum
acceptable input length for the model if that argument is not provided.
False
or
'do_not_pad'
(default): No padding (i.e., can output a batch with sequences of different
lengths).
truncation
(
bool
, defaults to
False
) —
Whether to truncate sequences at the maximum length. Has no effect if tokenize is
False
.
max_length
(
int
,
optional
) —
Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is
False
. If
not specified, the tokenizer’s
max_length
attribute will be used as a default.
return_tensors
(
str
or
TensorType
,
optional
) —
If set, will return tensors of a particular framework. Has no effect if tokenize is
False
. Acceptable
values are:
'pt'
: Return PyTorch
torch.Tensor
objects.
'np'
: Return NumPy
np.ndarray
objects.
return_dict
(
bool
, defaults to
True
) —
Whether to return a dictionary with named outputs. Has no effect if tokenize is
False
.
tokenizer_kwargs
(
dict[str -- Any]
,
optional
): Additional kwargs to pass to the tokenizer.
return_assistant_tokens_mask
(
bool
, defaults to
False
) —
Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
the mask will contain 1. For user and system tokens, the mask will contain 0.
This functionality is only available for chat templates that support it via the
{% generation %}
keyword.
*
*kwargs
— Additional kwargs to pass to the template renderer. Will be accessible by the chat template.
Returns
Union[list[int], Dict]
A list of token ids representing the tokenized chat so far, including control tokens. This
output is ready to pass to the model, either directly or via methods like
generate()
. If
return_dict
is
set, will return a dict of tokenizer outputs instead.
Converts a list of dictionaries with
"role"
and
"content"
keys to a list of token
ids. This method is intended for use with chat models, and will read the tokenizer’s chat_template attribute to
determine the format and control tokens to use when converting.
batch_decode
<
source
>
(
sequences
: list[int] | list[list[int]] | np.ndarray | torch.Tensor
skip_special_tokens
: bool = False
clean_up_tokenization_spaces
: bool | None = None
**kwargs
)
→
list[str]
Parameters
sequences
(
Union[list[int], list[list[int]], np.ndarray, torch.Tensor]
) —
List of tokenized input ids. Can be obtained using the
__call__
method.
skip_special_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not to remove special tokens in the decoding.
clean_up_tokenization_spaces
(
bool
,
optional
) —
Whether or not to clean up the tokenization spaces. If
None
, will default to
self.clean_up_tokenization_spaces
.
kwargs
(additional keyword arguments,
optional
) —
Will be passed to the underlying model specific decode method.
Returns
list[str]
The list of decoded sentences.
Convert a list of lists of token ids into a list of strings by calling decode.
This method is provided for backwards compatibility. The
decode
method now handles batched input natively,
so you can use
decode
directly instead of
batch_decode
.
decode
<
source
>
(
token_ids
: int | list[int] | list[list[int]] | np.ndarray | torch.Tensor
skip_special_tokens
: bool = False
**kwargs
)
→
Union[str, list[str]]
Parameters
token_ids
(
Union[int, list[int], list[list[int]], np.ndarray, torch.Tensor]
) —
A single sequence or a batch (list of sequences) of tokenized input ids. Can be obtained using the
__call__
method.
skip_special_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not to remove special tokens in the decoding.
kwargs
(additional keyword arguments,
optional
) —
Will be passed to the underlying model specific decode method.
Returns
Union[str, list[str]]
The decoded string for a single sequence, or a list of decoded strings for a
batch of sequences.
Converts a sequence of ids into a string, or a list of sequences into a list of strings,
using the tokenizer and vocabulary with options to remove special tokens and clean up
tokenization spaces.
Similar to doing
self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))
.
encode
<
source
>
(
text
: TextInput | PreTokenizedInput | EncodedInput
text_pair
: TextInput | PreTokenizedInput | EncodedInput | None = None
add_special_tokens
: bool = True
padding
: bool | str | PaddingStrategy = False
truncation
: bool | str | TruncationStrategy | None = None
max_length
: int | None = None
stride
: int = 0
padding_side
: str | None = None
return_tensors
: str | TensorType | None = None
**kwargs
)
→
list[int]
,
torch.Tensor
, or
np.ndarray
Parameters
text
(
str
,
list[str]
or
list[int]
) —
The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
tokenize
method) or a list of integers (tokenized string ids using the
convert_tokens_to_ids
method).
text_pair
(
str
,
list[str]
or
list[int]
,
optional
) —
Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
the
tokenize
method) or a list of integers (tokenized string ids using the
convert_tokens_to_ids
method).
add_special_tokens
(
bool
,
optional
, defaults to
True
) —
Whether or not to add special tokens when encoding the sequences. This will use the underlying
PretrainedTokenizerBase.build_inputs_with_special_tokens
function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add
bos
or
eos
tokens
automatically.
padding
(
bool
,
str
or
PaddingStrategy
,
optional
, defaults to
False
) —
Activates and controls padding. Accepts the following values:
True
or
'longest'
: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).
'max_length'
: Pad to a maximum length specified with the argument
max_length
or to the maximum
acceptable input length for the model if that argument is not provided.
False
or
'do_not_pad'
(default): No padding (i.e., can output a batch with sequences of different
lengths).
truncation
(
bool
,
str
or
TruncationStrategy
,
optional
, defaults to
False
) —
Activates and controls truncation. Accepts the following values:
True
or
'longest_first'
: Truncate to a maximum length specified with the argument
max_length
or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.
'only_first'
: Truncate to a maximum length specified with the argument
max_length
or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
'only_second'
: Truncate to a maximum length specified with the argument
max_length
or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
False
or
'do_not_truncate'
(default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).
max_length
(
int
,
optional
) —
Controls the maximum length to use by one of the truncation/padding parameters.
If left unset or set to
None
, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.
stride
(
int
,
optional
, defaults to 0) —
If set to a number along with
max_length
, the overflowing tokens returned when
return_overflowing_tokens=True
will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.
is_split_into_words
(
bool
,
optional
, defaults to
False
) —
Whether or not the input is already pre-tokenized (e.g., split into words). If set to
True
, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.
pad_to_multiple_of
(
int
,
optional
) —
If set will pad the sequence to a multiple of the provided value. Requires
padding
to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
>= 7.5
(Volta).
padding_side
(
str
,
optional
) —
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
return_tensors
(
str
or
TensorType
,
optional
) —
If set, will return tensors instead of list of python integers. Acceptable values are:
'pt'
: Return PyTorch
torch.Tensor
objects.
'np'
: Return Numpy
np.ndarray
objects.
*
*kwargs
— Passed along to the
.tokenize()
method.
Returns
list[int]
,
torch.Tensor
, or
np.ndarray
The tokenized ids of the text.
Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.
Same as doing
self.convert_tokens_to_ids(self.tokenize(text))
.
push_to_hub
<
source
>
(
repo_id
: str
commit_message
: str | None = None
commit_description
: str | None = None
private
: bool | None = None
token
: bool | str | None = None
revision
: str | None = None
create_pr
: bool = False
max_shard_size
: int | str | None = '50GB'
tags
: list[str] | None = None
)
Parameters
repo_id
(
str
) —
The name of the repository you want to push your tokenizer to. It should contain your organization name
when pushing to a given organization.
commit_message
(
str
,
optional
) —
Message to commit while pushing. Will default to
"Upload tokenizer"
.
commit_description
(
str
,
optional
) —
The description of the commit that will be created
private
(
bool
,
optional
) —
Whether to make the repo private. If
None
(default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
token
(
bool
or
str
,
optional
) —
The token to use as HTTP bearer authorization for remote files. If
True
(default), will use the token generated
when running
hf auth login
(stored in
~/.huggingface
).
revision
(
str
,
optional
) —
Branch to push the uploaded files to.
create_pr
(
bool
,
optional
, defaults to
False
) —
Whether or not to create a PR with the uploaded files or directly commit.
max_shard_size
(
int
or
str
,
optional
, defaults to
"50GB"
) —
Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
will then be each of size lower than this size. If expressed as a string, needs to be digits followed
by a unit (like
"5MB"
).
tags
(
list[str]
,
optional
) —
List of tags to push on the Hub.
Upload the tokenizer files to the 🤗 Model Hub.
Examples:
Copied
from
transformers
import
AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
"google-bert/bert-base-cased"
)
# Push the tokenizer to your namespace with the name "my-finetuned-bert".
tokenizer.push_to_hub(
"my-finetuned-bert"
)
# Push the tokenizer to an organization with the name "my-finetuned-bert".
tokenizer.push_to_hub(
"huggingface/my-finetuned-bert"
)
convert_to_native_format
<
source
>
(
trust_remote_code
= False
**kwargs
)
Build a
tokenizers.Tokenizer
backend from the available serialization files (tokenizer.json, sentencepiece
models, tekken.json, vocab/merges).
get_added_vocab
<
source
>
(
)
→
dict[str, int]
Returns
dict[str, int]
The added tokens.
Returns the added tokens in the vocabulary as a dictionary of token to index.
num_special_tokens_to_add
<
source
>
(
pair
: bool = False
)
→
int
Parameters
pair
(
bool
,
optional
, defaults to
False
) —
Whether the number of added tokens should be computed in the case of a sequence pair or a single
sequence.
Returns
int
Number of special tokens added to sequences.
Returns the number of added tokens when encoding a sequence with special tokens.
This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
this inside your training loop.
save_pretrained
<
source
>
(
save_directory
: str | os.PathLike
legacy_format
: bool | None = None
filename_prefix
: str | None = None
push_to_hub
: bool = False
save_format
: str | None = None
**kwargs
)
→
A tuple of
str
Parameters
save_format
(
str
,
optional
) —
"mistral"
to save as a native
tekken.json
by copying the original
file (requires the original tekken vocabulary file to be available). The name is
normalized to
tekken.json
on save, and the instantiated tokenizer must still match
it exactly.
"hf"
or
None
for the default HuggingFace format.
kwargs
(
dict[str, Any]
,
optional
) —
Additional key word arguments passed along to the
push_to_hub()
method.
Returns
A tuple of
str
The files saved.
Save the full tokenizer state.
Extends
save_pretrained()
with
save_format
support, for tokenizers converted from a native vocabulary file
(e.g. Mistral’s
tekken.json
). See the base method for
legacy_format
,
filename_prefix
, and
push_to_hub
.
set_truncation_and_padding
<
source
>
(
padding_strategy
: PaddingStrategy
truncation_strategy
: TruncationStrategy
max_length
: int
stride
: int
pad_to_multiple_of
: int | None
padding_side
: str | None
)
Parameters
padding_strategy
(
PaddingStrategy
) —
The kind of padding that will be applied to the input
truncation_strategy
(
TruncationStrategy
) —
The kind of truncation that will be applied to the input
max_length
(
int
) —
The maximum size of a sequence.
stride
(
int
) —
The stride to use when handling overflow.
pad_to_multiple_of
(
int
,
optional
) —
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability
>= 7.5
(Volta).
padding_side
(
str
,
optional
) —
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
library) and restore the tokenizer settings afterwards.
The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
section.
train_new_from_iterator
<
source
>
(
text_iterator
vocab_size
length
= None
new_special_tokens
= None
special_tokens_map
= None
**kwargs
)
→
PreTrainedTokenizerFast
Parameters
text_iterator
(generator of
list[str]
) —
The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts
if you have everything in memory.
vocab_size
(
int
) —
The size of the vocabulary you want for your tokenizer.
length
(
int
,
optional
) —
The total number of sequences in the iterator. This is used to provide meaningful progress tracking
new_special_tokens
(list of
str
or
AddedToken
,
optional
) —
A list of new special tokens to add to the tokenizer you are training.
special_tokens_map
(
dict[str, str]
,
optional
) —
If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special
token name to new special token name in this argument.
kwargs
(
dict[str, Any]
,
optional
) —
Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.
Returns
PreTrainedTokenizerFast
A new tokenizer of the same type as the original one, trained on
text_iterator
.
Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
as the current one.
update_post_processor
<
source
>
(
)
Updates the underlying post processor with the current
bos_token
and
eos_token
.
PythonBackend
class
transformers.
PythonBackend
<
source
>
(
**kwargs
)
Parameters
model_max_length
(
int
,
optional
) —
The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
loaded with
from_pretrained()
, this will be set to the
value stored for the associated model in
max_model_input_sizes
(see above). If no value is provided, will
default to VERY_LARGE_INTEGER (
int(1e30)
).
padding_side
(
str
,
optional
) —
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
truncation_side
(
str
,
optional
) —
The side on which the model should have truncation applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
chat_template
(
str
,
optional
) —
A Jinja template string that will be used to format lists of chat messages. See
https://huggingface.co/docs/transformers/chat_templating
for a full description.
model_input_names
(
list[string]
,
optional
) —
The list of inputs accepted by the forward pass of the model (like
"token_type_ids"
or
"attention_mask"
). Default value is picked from the class attribute of the same name.
bos_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the beginning of a sentence.
eos_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the end of a sentence.
unk_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing an out-of-vocabulary token.
sep_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token separating two different sentences in the same input (used by BERT for instance).
pad_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
attention mechanisms or loss computation.
cls_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the class of the input (used by BERT for instance).
mask_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing a masked token (used by masked-language modeling pretraining objectives, like
BERT). Will be associated to
self.mask_token
and
self.mask_token_id
.
extra_special_tokens
(list of
str
or
tokenizers.AddedToken
,
optional
) —
A list of extra model-specific special tokens. Add them here to ensure they are skipped when decoding with
skip_special_tokens
is set to True. If they are not part of the vocabulary, they will be added at the end
of the vocabulary.
split_special_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not the special tokens should be split during the tokenization process. Passing will affect the
internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
<s>
is the
bos_token
, then
tokenizer.tokenize("<s>") = ['<s>
]. Otherwise, if
split_special_tokens=True
, then
tokenizer.tokenize("<s>")
will be give
['<','s', '>']
.
Base class for all slow tokenizers.
Inherits from
PreTrainedTokenizerBase
.
Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
pretrained tokenizers as well as adding tokens to the vocabulary.
This class also contain the added tokens in a unified way on top of all tokenizers so we don’t have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).
Class attributes (overridden by derived classes)
vocab_files_names
(
dict[str, str]
) — A dictionary with, as keys, the
__init__
keyword name of each
vocabulary file required by the model, and as associated values, the filename for saving the associated file
(string).
pretrained_vocab_files_map
(
dict[str, dict[str, str]]
) — A dictionary of dictionaries, with the
high-level keys being the
__init__
keyword name of each vocabulary file required by the model, the
low-level being the
short-cut-names
of the pretrained models with, as associated values, the
url
to the
associated pretrained vocabulary file.
model_input_names
(
list[str]
) — A list of inputs expected in the forward pass of the model.
padding_side
(
str
) — The default value for the side on which the model should have padding applied.
Should be
'right'
or
'left'
.
truncation_side
(
str
) — The default value for the side on which the model should have truncation
applied. Should be
'right'
or
'left'
.
build_inputs_with_special_tokens
<
source
>
(
token_ids_0
: list
token_ids_1
: list[int] | None = None
)
→
list[int]
Parameters
token_ids_0
(
list[int]
) —
List of IDs to which the special tokens will be added.
token_ids_1
(
list[int]
,
optional
) —
Optional second list of IDs for sequence pairs.
Returns
list[int]
List of input IDs with the appropriate special tokens.
Build model inputs from a sequence or a pair of sequences by adding special tokens.
This method dynamically builds inputs based on the tokenizer’s
special_tokens_pattern
:
"none"
: No special tokens
"cls_sep"
: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
"eos"
: seq0 [EOS] or seq0 [EOS] seq1 [EOS]
"bos"
: [BOS] seq0 or [BOS] seq0 [BOS] seq1
"bos_eos"
: [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
"cls_double_sep"
: [CLS] seq0 [SEP] or [CLS] seq0 [SEP][SEP] seq1 [SEP]
"prefix_suffix"
:
<prefix_tokens> seq0 [seq1] <suffix_tokens>
(custom prefix/suffix stored on the tokenizer)
create_token_type_ids_from_sequences
<
source
>
(
token_ids_0
: list
token_ids_1
: list[int] | None = None
)
→
list[int]
Parameters
token_ids_0
(
list[int]
) —
List of IDs.
token_ids_1
(
list[int]
,
optional
) —
Optional second list of IDs for sequence pairs.
Returns
list[int]
Token type IDs according to the configured pattern.
Create a mask from the two sequences passed to be used in a sequence-pair classification task.
This method dynamically builds the token type IDs based on the tokenizer’s configuration attributes:
token_type_ids_pattern
: Pattern to use (“all_zeros” or “bert_style”)
token_type_ids_include_special_tokens
: Whether to account for special tokens in length calculation
Examples:
Copied
# All zeros pattern (default, used by RoBERTa, BART, etc.)
tokenizer.token_type_ids_pattern =
"all_zeros"
# Returns: [0, 0, 0, ...] for both sequences
# BERT-style pattern (first sequence gets 0s, second gets 1s)
tokenizer.token_type_ids_pattern =
"bert_style"
# Returns: [0, 0, 0, ..., 1, 1, 1, ...] for sequence pairs
get_added_vocab
<
source
>
(
)
→
dict[str, int]
Returns
dict[str, int]
The added tokens.
Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
something we should change.
get_special_tokens_mask
<
source
>
(
token_ids_0
: list
token_ids_1
: list | None = None
already_has_special_tokens
: bool = False
)
→
A list of integers in the range [0, 1]
Parameters
token_ids_0
(
list[int]
) —
List of ids of the first sequence.
token_ids_1
(
list[int]
,
optional
) —
List of ids of the second sequence.
already_has_special_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not the token list is already formatted with special tokens for the model.
Returns
A list of integers in the range [0, 1]
1 for a special token, 0 for a sequence token.
Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer
prepare_for_model
or
encode_plus
methods.
This method dynamically builds the special tokens mask based on the tokenizer’s
special_tokens_pattern
:
"none"
: No special tokens (default, returns all 0s)
"cls_sep"
: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
"eos"
: seq0 [EOS] or seq0 [EOS] seq1 [EOS]
"bos"
: [BOS] seq0 or [BOS] seq0 [BOS] seq1
"bos_eos"
: [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
"cls_double_sep"
: [CLS] seq0 [SEP] or [CLS] seq0 [SEP][SEP] seq1 [SEP]
"prefix_suffix"
:
<prefix_tokens> seq0 [seq1] <suffix_tokens>
num_special_tokens_to_add
<
source
>
(
pair
: bool = False
)
→
int
Parameters
pair
(
bool
,
optional
, defaults to
False
) —
Whether the number of added tokens should be computed in the case of a sequence pair or a single
sequence.
Returns
int
Number of special tokens added to sequences.
Returns the number of added tokens when encoding a sequence with special tokens.
This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
this inside your training loop.
prepare_for_model
<
source
>
(
ids
: list
pair_ids
: list[int] | None = None
add_special_tokens
: bool = True
padding
: bool | str | transformers.utils.generic.PaddingStrategy = False
truncation
: bool | str | transformers.tokenization_utils_base.TruncationStrategy = False
max_length
: int | None = None
stride
: int = 0
pad_to_multiple_of
: int | None = None
padding_side
: str | None = None
return_tensors
: str | transformers.utils.generic.TensorType | None = None
return_token_type_ids
: bool | None = None
return_attention_mask
: bool | None = None
return_overflowing_tokens
: bool = False
return_special_tokens_mask
: bool = False
return_length
: bool = False
verbose
: bool = True
prepend_batch_axis
: bool = False
**kwargs
)
Parameters
ids
— Tokenized input ids of the first sequence.
pair_ids
— Tokenized input ids of the second sequence (optional).
Prepares a sequence of input ids so it can be used by the model. Adds special tokens, truncates, and pads.
prepare_for_tokenization
<
source
>
(
text
: str
is_split_into_words
: bool = False
**kwargs
)
→
tuple[str, dict[str, Any]]
Parameters
text
(
str
) —
The text to prepare.
is_split_into_words
(
bool
,
optional
, defaults to
False
) —
Whether or not the input is already pre-tokenized (e.g., split into words). If set to
True
, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.
kwargs
(
dict[str, Any]
,
optional
) —
Keyword arguments to use for the tokenization.
Returns
tuple[str, dict[str, Any]]
The prepared text and the unused kwargs.
Performs any necessary transformations before tokenization.
This method should pop the arguments from kwargs and return the remaining
kwargs
as well. We test the
kwargs
at the end of the encoding process to be sure all the arguments have been used.
save_vocabulary
<
source
>
(
save_directory
: str
filename_prefix
: str | None = None
)
→
tuple[str, ...]
Parameters
save_directory
(
str
) —
The directory in which to save the vocabulary.
filename_prefix
(
str
,
optional
) —
An optional prefix to add to the named of the saved files.
Returns
tuple[str, ...]
Paths to the files saved, or empty tuple if no files saved.
Default implementation for common vocabulary saving patterns.
Saves self.encoder/self.vocab as JSON, optionally with self.bpe_ranks as merges.
Returns empty tuple if no vocabulary exists.
Override this method if your tokenizer needs custom saving logic (e.g., SentencePiece models,
multiple vocabulary files, or special file formats).
tokenize
<
source
>
(
text
: str
**kwargs
)
Parameters
text
— The sequence to be encoded.
*
*kwargs
— Passed along to the model-specific
prepare_for_tokenization
preprocessing method.
Converts a string into a sequence of tokens, using the tokenizer.
truncate_sequences
<
source
>
(
ids
: list
pair_ids
: list[int] | None = None
num_tokens_to_remove
: int = 0
truncation_strategy
: str | transformers.tokenization_utils_base.TruncationStrategy = 'longest_first'
stride
: int = 0
)
Truncates sequences according to the specified strategy.
TokenizersBackend
class
transformers.
TokenizersBackend
<
source
>
(
*args
**kwargs
)
Parameters
model_max_length
(
int
,
optional
) —
The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
loaded with
from_pretrained()
, this will be set to the
value stored for the associated model in
max_model_input_sizes
(see above). If no value is provided, will
default to VERY_LARGE_INTEGER (
int(1e30)
).
padding_side
(
str
,
optional
) —
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
truncation_side
(
str
,
optional
) —
The side on which the model should have truncation applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
chat_template
(
str
,
optional
) —
A Jinja template string that will be used to format lists of chat messages. See
https://huggingface.co/docs/transformers/chat_templating
for a full description.
model_input_names
(
list[string]
,
optional
) —
The list of inputs accepted by the forward pass of the model (like
"token_type_ids"
or
"attention_mask"
). Default value is picked from the class attribute of the same name.
bos_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the beginning of a sentence.
eos_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the end of a sentence.
unk_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing an out-of-vocabulary token.
sep_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token separating two different sentences in the same input (used by BERT for instance).
pad_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
attention mechanisms or loss computation.
cls_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the class of the input (used by BERT for instance).
mask_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing a masked token (used by masked-language modeling pretraining objectives, like
BERT). Will be associated to
self.mask_token
and
self.mask_token_id
.
extra_special_tokens
(list of
str
or
tokenizers.AddedToken
,
optional
) —
A list of extra model-specific special tokens. Add them here to ensure they are skipped when decoding with
skip_special_tokens
is set to True. If they are not part of the vocabulary, they will be added at the end
of the vocabulary.
split_special_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not the special tokens should be split during the tokenization process. Passing will affect the
internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
<s>
is the
bos_token
, then
tokenizer.tokenize("<s>") = ['<s>
]. Otherwise, if
split_special_tokens=True
, then
tokenizer.tokenize("<s>")
will be give
['<','s', '>']
.
tokenizer_object
(
tokenizers.Tokenizer
) —
A
tokenizers.Tokenizer
object from 🤗 tokenizers to instantiate from. See
Using tokenizers from 🤗
tokenizers
for more information.
tokenizer_file
(
str
) —
A path to a local JSON file representing a previously serialized
tokenizers.Tokenizer
object from 🤗
tokenizers.
Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).
Inherits from
PreTrainedTokenizerBase
.
Handles all the shared methods for tokenization and special tokens, as well as methods for
downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.
This class also contains the added tokens in a unified way on top of all tokenizers so we don’t have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).
Class attributes (overridden by derived classes)
vocab_files_names
(
dict[str, str]
) — A dictionary with, as keys, the
__init__
keyword name of each
vocabulary file required by the model, and as associated values, the filename for saving the associated file
(string).
pretrained_vocab_files_map
(
dict[str, dict[str, str]]
) — A dictionary of dictionaries, with the
high-level keys being the
__init__
keyword name of each vocabulary file required by the model, the
low-level being the
short-cut-names
of the pretrained models with, as associated values, the
url
to the
associated pretrained vocabulary file.
model_input_names
(
list[str]
) — A list of inputs expected in the forward pass of the model.
padding_side
(
str
) — The default value for the side on which the model should have padding applied.
Should be
'right'
or
'left'
.
truncation_side
(
str
) — The default value for the side on which the model should have truncation
applied. Should be
'right'
or
'left'
.
convert_to_native_format
<
source
>
(
trust_remote_code
= False
**kwargs
)
Build a
tokenizers.Tokenizer
backend from the available serialization files (tokenizer.json, sentencepiece
models, tekken.json, vocab/merges).
get_added_vocab
<
source
>
(
)
→
dict[str, int]
Returns
dict[str, int]
The added tokens.
Returns the added tokens in the vocabulary as a dictionary of token to index.
num_special_tokens_to_add
<
source
>
(
pair
: bool = False
)
→
int
Parameters
pair
(
bool
,
optional
, defaults to
False
) —
Whether the number of added tokens should be computed in the case of a sequence pair or a single
sequence.
Returns
int
Number of special tokens added to sequences.
Returns the number of added tokens when encoding a sequence with special tokens.
This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
this inside your training loop.
save_pretrained
<
source
>
(
save_directory
: str | os.PathLike
legacy_format
: bool | None = None
filename_prefix
: str | None = None
push_to_hub
: bool = False
save_format
: str | None = None
**kwargs
)
→
A tuple of
str
Parameters
save_format
(
str
,
optional
) —
"mistral"
to save as a native
tekken.json
by copying the original
file (requires the original tekken vocabulary file to be available). The name is
normalized to
tekken.json
on save, and the instantiated tokenizer must still match
it exactly.
"hf"
or
None
for the default HuggingFace format.
kwargs
(
dict[str, Any]
,
optional
) —
Additional key word arguments passed along to the
push_to_hub()
method.
Returns
A tuple of
str
The files saved.
Save the full tokenizer state.
Extends
save_pretrained()
with
save_format
support, for tokenizers converted from a native vocabulary file
(e.g. Mistral’s
tekken.json
). See the base method for
legacy_format
,
filename_prefix
, and
push_to_hub
.
set_truncation_and_padding
<
source
>
(
padding_strategy
: PaddingStrategy
truncation_strategy
: TruncationStrategy
max_length
: int
stride
: int
pad_to_multiple_of
: int | None
padding_side
: str | None
)
Parameters
padding_strategy
(
PaddingStrategy
) —
The kind of padding that will be applied to the input
truncation_strategy
(
TruncationStrategy
) —
The kind of truncation that will be applied to the input
max_length
(
int
) —
The maximum size of a sequence.
stride
(
int
) —
The stride to use when handling overflow.
pad_to_multiple_of
(
int
,
optional
) —
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability
>= 7.5
(Volta).
padding_side
(
str
,
optional
) —
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
library) and restore the tokenizer settings afterwards.
The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
section.
train_new_from_iterator
<
source
>
(
text_iterator
vocab_size
length
= None
new_special_tokens
= None
special_tokens_map
= None
**kwargs
)
→
PreTrainedTokenizerFast
Parameters
text_iterator
(generator of
list[str]
) —
The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts
if you have everything in memory.
vocab_size
(
int
) —
The size of the vocabulary you want for your tokenizer.
length
(
int
,
optional
) —
The total number of sequences in the iterator. This is used to provide meaningful progress tracking
new_special_tokens
(list of
str
or
AddedToken
,
optional
) —
A list of new special tokens to add to the tokenizer you are training.
special_tokens_map
(
dict[str, str]
,
optional
) —
If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special
token name to new special token name in this argument.
kwargs
(
dict[str, Any]
,
optional
) —
Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.
Returns
PreTrainedTokenizerFast
A new tokenizer of the same type as the original one, trained on
text_iterator
.
Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
as the current one.
update_post_processor
<
source
>
(
)
Updates the underlying post processor with the current
bos_token
and
eos_token
.
SentencePieceBackend
class
transformers.
SentencePieceBackend
<
source
>
(
**kwargs
)
Parameters
model_max_length
(
int
,
optional
) —
The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
loaded with
from_pretrained()
, this will be set to the
value stored for the associated model in
max_model_input_sizes
(see above). If no value is provided, will
default to VERY_LARGE_INTEGER (
int(1e30)
).
padding_side
(
str
,
optional
) —
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
truncation_side
(
str
,
optional
) —
The side on which the model should have truncation applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
chat_template
(
str
,
optional
) —
A Jinja template string that will be used to format lists of chat messages. See
https://huggingface.co/docs/transformers/chat_templating
for a full description.
model_input_names
(
list[string]
,
optional
) —
The list of inputs accepted by the forward pass of the model (like
"token_type_ids"
or
"attention_mask"
). Default value is picked from the class attribute of the same name.
bos_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the beginning of a sentence.
eos_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the end of a sentence.
unk_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing an out-of-vocabulary token.
sep_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token separating two different sentences in the same input (used by BERT for instance).
pad_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
attention mechanisms or loss computation.
cls_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing the class of the input (used by BERT for instance).
mask_token
(
str
or
tokenizers.AddedToken
,
optional
) —
A special token representing a masked token (used by masked-language modeling pretraining objectives, like
BERT). Will be associated to
self.mask_token
and
self.mask_token_id
.
extra_special_tokens
(list of
str
or
tokenizers.AddedToken
,
optional
) —
A list of extra model-specific special tokens. Add them here to ensure they are skipped when decoding with
skip_special_tokens
is set to True. If they are not part of the vocabulary, they will be added at the end
of the vocabulary.
split_special_tokens
(
bool
,
optional
, defaults to
False
) —
Whether or not the special tokens should be split during the tokenization process. Passing will affect the
internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
<s>
is the
bos_token
, then
tokenizer.tokenize("<s>") = ['<s>
]. Otherwise, if
split_special_tokens=True
, then
tokenizer.tokenize("<s>")
will be give
['<','s', '>']
.
Base class for SentencePiece-based tokenizers that load from sentencepiece.model files.
Inherits from
PreTrainedTokenizer
.
Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
pretrained tokenizers as well as adding tokens to the vocabulary.
This class also contain the added tokens in a unified way on top of all tokenizers so we don’t have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).
Class attributes (overridden by derived classes)
vocab_files_names
(
dict[str, str]
) — A dictionary with, as keys, the
__init__
keyword name of each
vocabulary file required by the model, and as associated values, the filename for saving the associated file
(string).
pretrained_vocab_files_map
(
dict[str, dict[str, str]]
) — A dictionary of dictionaries, with the
high-level keys being the
__init__
keyword name of each vocabulary file required by the model, the
low-level being the
short-cut-names
of the pretrained models with, as associated values, the
url
to the
associated pretrained vocabulary file.
model_input_names
(
list[str]
) — A list of inputs expected in the forward pass of the model.
padding_side
(
str
) — The default value for the side on which the model should have padding applied.
Should be
'right'
or
'left'
.
truncation_side
(
str
) — The default value for the side on which the model should have truncation
applied. Should be
'right'
or
'left'
.
convert_tokens_to_string
<
source
>
(
tokens
: list
)
Converts a sequence of tokens (string) in a single string.
get_vocab
<
source
>
(
)
Returns vocab as a dict
save_vocabulary
<
source
>
(
save_directory
: str
filename_prefix
: str | None = None
)
→
tuple(str)
Parameters
save_directory
(
str
) —
The directory in which to save the vocabulary.
filename_prefix
(
str
,
optional
) —
An optional prefix to add to the named of the saved files.
Returns
tuple(str)
Paths to the files saved.
Save the sentencepiece vocabulary (copy original file) to a directory.
BatchEncoding
class
transformers.
BatchEncoding
<
source
>
(
data
: dict[str, Any] | None = None
encoding
: EncodingFast | Sequence[EncodingFast] | None = None
tensor_type
: None | str | TensorType = None
prepend_batch_axis
: bool = False
n_sequences
: int | None = None
)
Parameters
data
(
dict
,
optional
) —
Dictionary of lists/arrays/tensors returned by the
__call__
/
encode_plus
/
batch_encode_plus
methods
(‘input_ids’, ‘attention_mask’, etc.).
encoding
(
tokenizers.Encoding
or
Sequence[tokenizers.Encoding]
,
optional
) —
If the tokenizer is a fast tokenizer which outputs additional information like mapping from word/character
space to token space the
tokenizers.Encoding
instance or list of instance (for batches) hold this
information.
tensor_type
(
Union[None, str, TensorType]
,
optional
) —
You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
initialization.
prepend_batch_axis
(
bool
,
optional
, defaults to
False
) —
Whether or not to add a batch axis when converting to tensors (see
tensor_type
above). Note that this
parameter has an effect if the parameter
tensor_type
is set,
otherwise has no effect
.
n_sequences
(
Optional[int]
,
optional
) —
The number of input sequences represented by each encoding (
None
for unknown,
1
for a single sequence and
2
for a pair of sequences).
Holds the output of the
call
()
,
~tokenization_utils_base.PreTrainedTokenizerBase.encode_plus
and
~tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus
methods (tokens, attention_masks, etc).
This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
utility methods to map from word/character space to token space.
char_to_token
<
source
>
(
batch_or_char_index
: int
char_index
: int | None = None
sequence_index
: int = 0
)
→
int
Parameters
batch_or_char_index
(
int
) —
Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
the character in the sequence
char_index
(
int
,
optional
) —
If a batch index is provided in
batch_or_char_index
, this can be the index of the character in the
sequence.
sequence_index
(
int
,
optional
, defaults to 0) —
If a pair of sequences is encoded in the batch this can be used to specify which sequence in the pair (0
or 1) the provided character index belongs to.
Returns
int
Index of the token, or None if the char index refers to a whitespace only token and whitespace is
trimmed with
trim_offsets=True
.
Get the index of the token in the encoded output comprising a character in the original string for a sequence
of the batch.
Can be called as:
self.char_to_token(char_index)
if batch size is 1
self.char_to_token(batch_index, char_index)
if batch size is greater or equal to 1
This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
words.
char_to_word
<
source
>
(
batch_or_char_index
: int
char_index
: int | None = None
sequence_index
: int = 0
)
→
int
Parameters
batch_or_char_index
(
int
) —
Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
the character in the original string.
char_index
(
int
,
optional
) —
If a batch index is provided in
batch_or_char_index
, this can be the index of the character in the
original string.
sequence_index
(
int
,
optional
, defaults to 0) —
If a pair of sequences is encoded in the batch this can be used to specify which sequence in the pair (0
or 1) the provided character index belongs to.
Returns
int
Index of the word containing the character.
Get the word in the original string corresponding to a character in the original string of a sequence of the
batch.
Can be called as:
self.char_to_word(char_index)
if batch size is 1
self.char_to_word(batch_index, char_index)
if batch size is greater than 1
This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
words.
convert_to_tensors
<
source
>
(
tensor_type
: str | TensorType | None = None
prepend_batch_axis
: bool = False
)
Parameters
tensor_type
(
str
or
TensorType
,
optional
) —
The type of tensors to use. If
str
, should be one of the values of the enum
TensorType
. If
None
, no modification is done.
prepend_batch_axis
(
bool
,
optional
, defaults to
False
) —
Whether or not to add the batch dimension during the conversion.
Convert the inner content to tensors.
sequence_ids
<
source
>
(
batch_index
: int = 0
)
→
list[Optional[int]]
Parameters
batch_index
(
int
,
optional
, defaults to 0) — The index to access in the batch.
Returns
list[Optional[int]]
A list indicating the sequence id corresponding to each token. Special tokens added
by the tokenizer are mapped to
None
and other tokens are mapped to the index of their corresponding
sequence.
Return a list mapping the tokens to the id of their original sentences:
None
for special tokens added around or between sequences,
0
for tokens corresponding to words in the first sequence,
1
for tokens corresponding to words in the second sequence when a pair of sequences was jointly
encoded.
to
<
source
>
(
device
: str | torch.device | int
non_blocking
: bool = False
)
→
BatchEncoding
Parameters
device
(
str
or
torch.device
or
int
) — The device to put the tensors on.
non_blocking
(
bool
) — Whether to perform the copy asynchronously.
Returns
BatchEncoding
The same instance after modification.
Send all values to device by calling
v.to(device, non_blocking=non_blocking)
(PyTorch only).
token_to_chars
<
source
>
(
batch_or_token_index
: int
token_index
: int | None = None
)
→
CharSpan
Parameters
batch_or_token_index
(
int
) —
Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
the token in the sequence.
token_index
(
int
,
optional
) —
If a batch index is provided in
batch_or_token_index
, this can be the index of the token or tokens in
the sequence.
Returns
CharSpan
Span of characters in the original string, or None, if the token
(e.g.
,
) doesn’t correspond to any chars in the origin string.
Get the character span corresponding to an encoded token in a sequence of the batch.
Character spans are returned as a
CharSpan
with:
start
— Index of the first character in the original string associated to the token.
end
— Index of the character following the last character in the original string associated to the
token.
Can be called as:
self.token_to_chars(token_index)
if batch size is 1
self.token_to_chars(batch_index, token_index)
if batch size is greater or equal to 1
token_to_sequence
<
source
>
(
batch_or_token_index
: int
token_index
: int | None = None
)
→
int
Parameters
batch_or_token_index
(
int
) —
Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
the token in the sequence.
token_index
(
int
,
optional
) —
If a batch index is provided in
batch_or_token_index
, this can be the index of the token in the
sequence.
Returns
int
Index of the input sequence containing the token (
0
for the first sequence or
1
for the second sequence of a pair).
Get the index of the sequence represented by the given token. In the general use case, this method returns
0
for a single sequence or the first sequence of a pair, and
1
for the second sequence of a pair
Can be called as:
self.token_to_sequence(token_index)
if batch size is 1
self.token_to_sequence(batch_index, token_index)
if batch size is greater than 1
This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
words are defined by the user). In this case it allows to easily associate encoded tokens with provided
tokenized words.
token_to_word
<
source
>
(
batch_or_token_index
: int
token_index
: int | None = None
)
→
int
Parameters
batch_or_token_index
(
int
) —
Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
the token in the sequence.
token_index
(
int
,
optional
) —
If a batch index is provided in
batch_or_token_index
, this can be the index of the token in the
sequence.
Returns
int
Index of the word in the input sequence.
Get the index of the word corresponding (i.e. comprising) to an encoded token in a sequence of the batch.
Can be called as:
self.token_to_word(token_index)
if batch size is 1
self.token_to_word(batch_index, token_index)
if batch size is greater than 1
This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
words are defined by the user). In this case it allows to easily associate encoded tokens with provided
tokenized words.
tokens
<
source
>
(
batch_index
: int = 0
)
→
list[str]
Parameters
batch_index
(
int
,
optional
, defaults to 0) — The index to access in the batch.
Returns
list[str]
The list of tokens at that index.
Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
integer indices) at a given batch index (only works for the output of a fast tokenizer).
word_ids
<
source
>
(
batch_index
: int = 0
)
→
list[Optional[int]]
Parameters
batch_index
(
int
,
optional
, defaults to 0) — The index to access in the batch.
Returns
list[Optional[int]]
A list indicating the word corresponding to each token. Special tokens added by the
tokenizer are mapped to
None
and other tokens are mapped to the index of their corresponding word
(several tokens will be mapped to the same word index if they are parts of that word).
Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.
word_to_chars
<
source
>
(
batch_or_word_index
: int
word_index
: int | None = None
sequence_index
: int = 0
)
→
CharSpan
Parameters
batch_or_word_index
(
int
) —
Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
the word in the sequence
word_index
(
int
,
optional
) —
If a batch index is provided in
batch_or_word_index
, this can be the index of the word in the
sequence.
sequence_index
(
int
,
optional
, defaults to 0) —
If a pair of sequences is encoded in the batch this can be used to specify which sequence in the pair (0
or 1) the provided word index belongs to.
Returns
CharSpan
Span of the associated character or characters in the string. CharSpan
is a NamedTuple with:
start: index of the first character associated to the word in the original string
end: index of the character following the last character associated to the word in the original
string
Get the character span in the original string corresponding to given word in a sequence of the batch.
Character spans are returned as a CharSpan NamedTuple with:
start: index of the first character in the original string
end: index of the character following the last character in the original string
Can be called as:
self.word_to_chars(word_index)
if batch size is 1
self.word_to_chars(batch_index, word_index)
if batch size is greater or equal to 1
word_to_tokens
<
source
>
(
batch_or_word_index
: int
word_index
: int | None = None
sequence_index
: int = 0
)
→
(
TokenSpan
,
optional
)
Parameters
batch_or_word_index
(
int
) —
Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
the word in the sequence.
word_index
(
int
,
optional
) —
If a batch index is provided in
batch_or_word_index
, this can be the index of the word in the
sequence.
sequence_index
(
int
,
optional
, defaults to 0) —
If a pair of sequences is encoded in the batch this can be used to specify which sequence in the pair (0
or 1) the provided word index belongs to.
Returns
(
TokenSpan
,
optional
)
Span of tokens in the encoded sequence. Returns
None
if no tokens correspond to the word. This can happen especially when the token is a special token
that has been used to format the tokenization. For example when we add a class token at the very beginning
of the tokenization.
Get the encoded token span corresponding to a word in a sequence of the batch.
Token spans are returned as a
TokenSpan
with:
start
— Index of the first token.
end
— Index of the token following the last token.
Can be called as:
self.word_to_tokens(word_index, sequence_index: int = 0)
if batch size is 1
self.word_to_tokens(batch_index, word_index, sequence_index: int = 0)
if batch size is greater or equal to
1
This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
words.
Update
on GitHub
←
Quantization
Trainer
→


---
