import os
import re
from abc import ABC, abstractmethod
from functools import partial
from inspect import signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
from huggingface_hub.hf_api import HfFolder
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
# from langchain.embeddings.
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (ConfigurableField, Runnable,
                                      RunnableConfig, RunnableParallel,
                                      RunnablePassthrough,
                                      RunnableSerializable, ensure_config)
from langchain_core.vectorstores import VectorStoreRetriever
from llama_cpp import Llama
from modules import RoPE, llama_cpp_python_hijack, shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger
from modules.text_generation import get_max_prompt_length

try:
    import llama_cpp
except:
    llama_cpp = None

try:
    import llama_cpp_cuda
except:
    llama_cpp_cuda = None

try:
    import llama_cpp_cuda_tensorcores
except:
    llama_cpp_cuda_tensorcores = None

token = "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"

HfFolder.save_token(token)

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")

def llama_cpp_lib():
    if shared.args.cpu and llama_cpp is not None:
        return llama_cpp
    elif shared.args.tensorcores and llama_cpp_cuda_tensorcores is not None:
        return llama_cpp_cuda_tensorcores
    elif llama_cpp_cuda is not None:
        return llama_cpp_cuda
    else:
        return llama_cpp


def ban_eos_logits_processor(eos_token, input_ids, logits):
    logits[eos_token] = -float('inf')
    return logits


def custom_token_ban_logits_processor(token_ids, input_ids, logits):
    for token_id in token_ids:
        logits[token_id] = -float('inf')

    return logits


RetrieverInput = str
RetrieverOutput = List[Document]
RetrieverLike = Runnable[RetrieverInput, RetrieverOutput]
RetrieverOutputLike = Runnable[Any, RetrieverOutput]

retriever1 = None
retriever2 = None
retriever3 = None

class RetrieverCustom(RunnableSerializable[RetrieverInput, RetrieverOutput], ABC):
    
    def __init__(self, r1 = VectorStoreRetriever, r2 = VectorStoreRetriever, r3 = VectorStoreRetriever):
        global retriever1, retriever2, retriever3
        retriever1 = r1
        retriever2 = r2
        retriever3 = r3
        
    def invoke(self, input, history):
        print("Within custom: ", history)
        global retriever1, retriever2, retriever3
        docs = retriever1.invoke(input)
        docs += retriever2.invoke(input)
        docs += retriever3.invoke(input)
        
        return docs

class LlamaCppModel:
    def __init__(self):
        self.initialized = False
        self.grammar_string = ''
        self.grammar = None

    def __del__(self):
        del self.model

    @classmethod
    def from_pretrained(self, path):

        Llama = llama_cpp_lib().Llama
        LlamaCache = llama_cpp_lib().LlamaCache

        result = self()
        cache_capacity = 0
        if shared.args.cache_capacity is not None:
            if 'GiB' in shared.args.cache_capacity:
                cache_capacity = int(re.sub('[a-zA-Z]', '', shared.args.cache_capacity)) * 1000 * 1000 * 1000
            elif 'MiB' in shared.args.cache_capacity:
                cache_capacity = int(re.sub('[a-zA-Z]', '', shared.args.cache_capacity)) * 1000 * 1000
            else:
                cache_capacity = int(shared.args.cache_capacity)

        if cache_capacity > 0:
            logger.info("Cache capacity is " + str(cache_capacity) + " bytes")

        if shared.args.tensor_split is None or shared.args.tensor_split.strip() == '':
            tensor_split_list = None
        else:
            tensor_split_list = [float(x) for x in shared.args.tensor_split.strip().split(",")]

        params = {
            'model_path': str(path),
            'n_ctx': shared.args.n_ctx,
            'n_threads': shared.args.threads or None,
            'n_threads_batch': shared.args.threads_batch or None,
            'n_batch': shared.args.n_batch,
            'use_mmap': not shared.args.no_mmap,
            'use_mlock': shared.args.mlock,
            'mul_mat_q': not shared.args.no_mul_mat_q,
            'numa': shared.args.numa,
            'n_gpu_layers': shared.args.n_gpu_layers,
            'rope_freq_base': RoPE.get_rope_freq_base(shared.args.alpha_value, shared.args.rope_freq_base),
            'tensor_split': tensor_split_list,
            'rope_freq_scale': 1.0 / shared.args.compress_pos_emb,
            'offload_kqv': not shared.args.no_offload_kqv,
            'split_mode': 1 if not shared.args.row_split else 2
        }

        result.model = Llama(**params)
        if cache_capacity > 0:
            result.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))

        result.initialise_vector_indexes()
        result.setup_rag_pipeline()
        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result

    def initialise_vector_indexes(self):
        em = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en-v1.5')
        self.vector_index = Neo4jVector.from_existing_graph(
            em,
            url=URI,
            username="neo4j",
            password="password",
            index_name='Protein',
            keyword_index_name='protein_name',
            node_label="protein",
            text_node_properties=['name', 'description', 'source'],
            embedding_node_property='embedding',
            search_type='hybrid',
        )

        self.vector_index2 = Neo4jVector.from_existing_graph(
            em,
            url=URI,
            username="neo4j",
            password="password",
            index_name='Drug',
            keyword_index_name='drug_description',
            node_label="drug",
            text_node_properties=['name','description', 'half_life', 'indication', 'mechanism_of_action', 'protein_binding', 'pharmacodynamics', 'state', 'atc_1', 'atc_2', 'atc_3', 'atc_4',
                'category', 'group', 'pathway', 'molecular_weight', 'tpsa', 'clogp'],
            embedding_node_property='embedding',
            search_type='hybrid'
        )

        self.vector_index3 = Neo4jVector.from_existing_graph(
            em,
            url=URI,
            username="neo4j",
            password="password",
            index_name='Disease',
            keyword_index_name='disease_description',
            node_label="disease",
            text_node_properties=['mondo_name', 'mondo_definition', 'umls_description', 'orphanet_definition', 'orphanet_prevalence', 'orphanet_epidemiology', 'orphanet_clinical_description', 'orphanet_management_and_treatment', 'mayo_symptoms', 'mayo_causes', 'mayo_risk_factors', 'mayo_complications', 'mayo_prevention', 'mayo_see_doc'],
            embedding_node_property='embedding',
            search_type='hybrid'
        )

    def setup_rag_pipeline(self):
        cq = """You are a medical chat assistant who answers user queries based on the context provided. 
            Keep the response precise and include all the requested information. Do not apologise or repeat responses. 
            In case you dont know the answer, Say `Apologies, I do not know the answer to this query.`

            Context: {context}
        """

        def get_chat_template(question):
            cqq = cq.format(context=question['context'])
            return [
                { "role": "system", "content": cqq},
                { "role": "user", "content": question['question']}
            ]

        def generate_with_configs(input):
            return self.model.create_chat_completion(input,
                max_tokens=512,
                temperature=0.1,
                top_p=0.1,
                min_p=0,
                typical_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                repeat_penalty=1.18,
                top_k=40,
                stream=True,
                seed=-1,
                tfs_z=1,
                mirostat_mode=0,
                mirostat_tau=1,
                mirostat_eta=0.1,
            )
    
        # contextualize_q_chain = get_chat_template | generate_with_configs | StrOutputParser()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        def contextualized_question(input: dict):
            return input["question"].replace("/", " ")
            if input.get("chat_history"):
                return contextualize_q_chain
            else:
                return input["question"]


        retriever1 = self.vector_index.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {"k":6, 'score_threshold': 1.0}
        )

        retriever2 = self.vector_index2.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {"k":6, 'score_threshold': 0.5}
        )

        retriever3 = self.vector_index3.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k":6, 'score_threshold': 0.5}
        )
        
        retriever1_advanced = self.vector_index.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {"k":6, 'score_threshold': 1.0}
        ).configurable_alternatives(
            ConfigurableField(id="strategy"),
            default_key="typical_rag",
            parent_strategy=retriever2,
            hypothetical_questions=retriever3,
            # summary_strategy=summary_vectorstore.as_retriever(),
        )
        
        retriever2_advanced = self.vector_index2.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {"k":6, 'score_threshold': 1.0}
        ).configurable_alternatives(
            ConfigurableField(id="strategy"),
            default_key="typical_rag",
            parent_strategy=retriever3,
            hypothetical_questions=retriever1,
            # summary_strategy=summary_vectorstore.as_retriever(),
        )
        
        retriever3_advanced = self.vector_index3.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {"k":6, 'score_threshold': 1.0}
        ).configurable_alternatives(
            ConfigurableField(id="strategy"),
            default_key="typical_rag",
            parent_strategy=retriever1,
            hypothetical_questions=retriever2,
            # summary_strategy=summary_vectorstore.as_retriever(),
        )

        
        self.rag_chain1 = (
            RunnablePassthrough.assign(
                context = contextualized_question | retriever1_advanced | format_docs
            )
            | get_chat_template 
            | generate_with_configs
        )
        self.rag_chain2 = (
            RunnablePassthrough.assign(
                context = contextualized_question | retriever2_advanced | format_docs
            )
            | get_chat_template 
            | generate_with_configs
        )
        self.rag_chain3 = (
            RunnablePassthrough.assign(
                context = contextualized_question | retriever3_advanced | format_docs
            )
            | get_chat_template 
            | generate_with_configs
        )
        
        retriever_full = RetrieverCustom(r1=retriever1_advanced, r2=retriever2_advanced, r3=retriever3_advanced)
        
        self.rag_full = (
            RunnablePassthrough.assign(
                context = contextualized_question | retriever_full | format_docs
            )
            | get_chat_template 
            | generate_with_configs
        )

    def encode(self, string):
        if type(string) is str:
            string = string.encode()

        return self.model.tokenize(string)

    def decode(self, ids, **kwargs):
        return self.model.detokenize(ids).decode('utf-8')

    def get_logits(self, tokens):
        self.model.reset()
        self.model.eval(tokens)
        logits = self.model._scores
        logits = np.expand_dims(logits, 0)  # batch dim is expected
        return torch.tensor(logits, dtype=torch.float32)

    def load_grammar(self, string):
        if string != self.grammar_string:
            self.grammar_string = string
            if string.strip() != '':
                self.grammar = llama_cpp_lib().LlamaGrammar.from_string(string)
            else:
                self.grammar = None

    def classify(self, prompt, state):
        prompt = """
            [INST] <<SYS>>
                You are doing a classification job. You will classify the given statement into one of the following categories drug, disease and protein based on the instruction provided. Classify into only one category primarily. Do not explain or respond with anything but the classified category.
            <</SYS>>
            
            Classify the given statement/question into either drug, disease or protein. `{prompt}`
            [/INST]
        """.format(prompt=prompt)
        LogitsProcessorList = llama_cpp_lib().LogitsProcessorList
        prompt = prompt if type(prompt) is str else prompt.decode()
        
        # Handle truncation
        prompt = self.encode(prompt)
        prompt = prompt[-get_max_prompt_length(state):]
        prompt = self.decode(prompt)

        self.load_grammar(state['grammar_string'])
        logit_processors = LogitsProcessorList()
        if state['ban_eos_token']:
            logit_processors.append(partial(ban_eos_logits_processor, self.model.token_eos()))

        if state['custom_token_bans']:
            to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
            if len(to_ban) > 0:
                logit_processors.append(partial(custom_token_ban_logits_processor, to_ban))

            
        print("################################ only the prompt -> ", prompt)
        completion_chunks = self.model.create_completion(
            prompt=prompt,
            max_tokens=state['max_new_tokens'],
            temperature=state['temperature'],
            top_p=state['top_p'],
            min_p=state['min_p'],
            typical_p=state['typical_p'],
            frequency_penalty=state['frequency_penalty'],
            presence_penalty=state['presence_penalty'],
            repeat_penalty=state['repetition_penalty'],
            top_k=state['top_k'],
            # stream=True,
            seed=int(state['seed']) if state['seed'] != -1 else None,
            tfs_z=state['tfs'],
            mirostat_mode=int(state['mirostat_mode']),
            mirostat_tau=state['mirostat_tau'],
            mirostat_eta=state['mirostat_eta'],
            logits_processor=logit_processors,
            grammar=self.grammar
        )

        print("completion_chunks ", completion_chunks)
        output = completion_chunks["choices"][0]["text"]

        dc = output.count("drug")
        pc = output.count("protein")
        dsc = output.count("disease")
        
        temp_max = 0
        if dc > pc:
            classified_obj = "drug"
            temp_max = dc
        else:
            classified_obj = "protein"
            temp_max = pc
        
        if dsc > temp_max:
            classified_obj = "disease"
            
        print("classified: ", output)
        return classified_obj
        
        
    def generate(self, prompt, state, callback=None):
        # Run the classifier model to get the classification from given prompt
        # internal_history = state.get("history", {}).get("internal", [])
        # classified_object = self.classify(state.get("textbox"), state)
        # print("LOLOLOLOLOL -----------------------", prompt, classified_object)
        LogitsProcessorList = llama_cpp_lib().LogitsProcessorList
        prompt = prompt if type(prompt) is str else prompt.decode()
        

        # Handle truncation
        prompt = self.encode(prompt)
        prompt = prompt[-get_max_prompt_length(state):]
        prompt = self.decode(prompt)

        self.load_grammar(state['grammar_string'])
        logit_processors = LogitsProcessorList()
        if state['ban_eos_token']:
            logit_processors.append(partial(ban_eos_logits_processor, self.model.token_eos()))

        if state['custom_token_bans']:
            to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
            if len(to_ban) > 0:
                logit_processors.append(partial(custom_token_ban_logits_processor, to_ban))

        # completion_chunks = self.model.create_completion(
        #     prompt=prompt,
        #     max_tokens=state['max_new_tokens'],
        #     temperature=state['temperature'],
        #     top_p=state['top_p'],
        #     min_p=state['min_p'],
        #     typical_p=state['typical_p'],
        #     frequency_penalty=state['frequency_penalty'],
        #     presence_penalty=state['presence_penalty'],
        #     repeat_penalty=state['repetition_penalty'],
        #     top_k=state['top_k'],
        #     stream=True,
        #     seed=int(state['seed']) if state['seed'] != -1 else None,
        #     tfs_z=state['tfs'],
        #     mirostat_mode=int(state['mirostat_mode']),
        #     mirostat_tau=state['mirostat_tau'],
        #     mirostat_eta=state['mirostat_eta'],
        #     logits_processor=logit_processors,
        #     grammar=self.grammar
        # )
            
        print("################################ only the prompt -> ", prompt)
        
        # if classified_object == "protein":
        #     completion_chunks = self.rag_chain1.invoke(
        #         {
        #             "question": prompt,
        #             "chat_history": []
        #         }
        #     )
        # elif classified_object == "drug":
        #     completion_chunks = self.rag_chain2.invoke(
        #         {
        #             "question": prompt,
        #             "chat_history": []
        #         }
        #     )
        # else:
        #     completion_chunks = self.rag_chain3.invoke(
        #         {
        #             "question": prompt,
        #             "chat_history": []
        #         }
        #     )
        
        completion_chunks = self.rag_full.invoke(
            {
                "question": prompt,
                "chat_history": []
            }
        )

        output = ""
        for completion_chunk in completion_chunks:
            if shared.stop_everything:
                break

            delta = completion_chunk["choices"][0]["delta"]
            # print("output text: ", text)
            text = ""
            if "content" in delta:
                text = delta["content"]
            
            output += text
            if callback:
                callback(text)

        return output

    def generate_with_streaming(self, *args, **kwargs):
        print("LOLOLOLOLOL")
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
