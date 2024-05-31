import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

filename = 'textfile.txt'
MODEL_SAVE_DIR = "model_save"
model_id = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
#model_id = "elyza/ELYZA-japanese-Llama-2-13b-instruct"
emb_model_id = "intfloat/multilingual-e5-large"

# Load and split documents
loader = TextLoader(filename, encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=300,
    chunk_overlap=20,
)
texts = text_splitter.split_documents(documents)
print(f"Number of text chunks: {len(texts)}")

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name=emb_model_id)
db = FAISS.from_documents(texts, embeddings)

# 一番類似するチャンクをいくつロードするかを変数kに設定できる
retriever = db.as_retriever(search_kwargs={"k": 3})

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_SAVE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
    cache_dir=MODEL_SAVE_DIR
).eval()

# Verify if CUDA is available and move the model to CUDA if possible
if torch.cuda.is_available():
    print("Using CUDA")
else:
    print("CUDA is not available, using CPU")

# Define prompt template
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "参考情報を元に、ユーザーからの質問にできるだけ正確に答えてください。"
text = "{context}\n=======\nユーザからの質問は次のとおりです。{question}"
template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
)
PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    template_format="f-string"
)

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(
        pipeline=pipe,
        # model_kwargs=dict(temperature=0.1, do_sample=True, repetition_penalty=1.1)
    ),
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    verbose=True,
)

while True:
    print("文章を入力してください")
    input_text = input(">>")
    start = time.time()
    inputs = template.format(context='', question=input_text)
    inputs = tokenizer(inputs, return_tensors='pt').to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
    print('RAG無し回答:', output)
    result = qa(input_text)
    print('RAG有り回答:', result['result'])
    print('='*10)
    print('ソース:', result['source_documents'])
    print('='*10)
    print("経過時間:", time.time() - start)
