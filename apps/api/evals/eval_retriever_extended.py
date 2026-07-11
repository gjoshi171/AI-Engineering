from api.agents.retrieval_generation import rag_pipeline

from langsmith import Client

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

from ragas.metrics.collections import Faithfulness, AnswerRelevancy

from qdrant_client import QdrantClient


ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")

openai_client = AsyncOpenAI()

ragas_llm = llm_factory("gpt-4.1-mini", client=openai_client, max_tokens=4000)
ragas_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")


def context_precision_id_based(run, example):

    retrieved_context_ids = {str(id) for id in run.outputs["retrieved_context_ids"]}
    reference_context_ids = {str(id) for id in example.outputs["reference_context_ids"]}

    score = len(retrieved_context_ids & reference_context_ids) / len(retrieved_context_ids) if retrieved_context_ids else 0.0

    return score


def context_recall_id_based(run, example):

    retrieved_context_ids = {str(id) for id in run.outputs["retrieved_context_ids"]}
    reference_context_ids = {str(id) for id in example.outputs["reference_context_ids"]}

    score = len(retrieved_context_ids & reference_context_ids) / len(reference_context_ids) if reference_context_ids else 0.0

    return score


def ragas_faithfulness(run, example):

    scorer = Faithfulness(llm=ragas_llm)

    result = scorer.score(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
        retrieved_contexts=run.outputs["retrieved_context"]
    )

    return result.value


def ragas_relevancy(run, example):

    scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    result = scorer.score(
        user_input=run.outputs["question"],
        response=run.outputs["answer"],
    )

    return result.value


dataset = list(ls_client.list_examples(dataset_name="rag-evaluation-dataset-extended"))

# Select only 10 examples
small_dataset = dataset[5:15]   # change to dataset[:10] if you want the first 10

print("Evalauating plain retriever")

results_plain = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client, top_k=10, hybrid=False, rerank=False),
    data=small_dataset,
    evaluators=[
        context_precision_id_based,
        context_recall_id_based,
        # ragas_faithfulness,
        # ragas_relevancy
    ],
    experiment_prefix="plain",
    max_concurrency=10
)

print("Evalauating hybrid retriever")

results_hybrid = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client, top_k=10, hybrid=True, rerank=False),
    data=small_dataset,
    evaluators=[
        context_precision_id_based,
        context_recall_id_based,
        # ragas_faithfulness,
        # ragas_relevancy
    ],
    experiment_prefix="hybrid",
    max_concurrency=10 # 10 runs in parallel
)

print("Evalauating hybrid retriever with reranking")

results_hybrid_rerank = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client, top_k=10, hybrid=True, rerank=True),
    data=small_dataset,
    evaluators=[
        context_precision_id_based,
        context_recall_id_based,
        # ragas_faithfulness,
        # ragas_relevancy
    ],
    experiment_prefix="hybrid-rerank",
    max_concurrency=1
)