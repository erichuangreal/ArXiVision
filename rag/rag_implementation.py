from dotenv import load_dotenv, find_dotenv
from rag.rag_class import RAGClass
import os

_env_path = find_dotenv(usecwd=True)
load_dotenv(_env_path, override=True)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("YOUR_") or api_key.strip() == "":
    raise RuntimeError(f"OPENAI_API_KEY missing or placeholder. Ensure a valid key is set in your .env (loaded from: {_env_path or 'not found'}).")
os.environ["OPENAI_API_KEY"] = api_key

if __name__ == "__main__":
    rag = RAGClass("processed_text/")

    # Load and process documents
    rag.load_documents()
    rag.split_documents()
    rag.create_vectorstore()
    rag.setup_retriever()
    rag.setup_qa_chain()

    # Answer a sample query
    rag.answer_query("What is AI?")

    # Evaluate using cosine similarity
    # The dev provides baseline queries and ground truths for testing

    queries = [
        # Paper 1: A dataset of rated conceptual arguments
        "What two properties define a conceptual question according to the paper?",
        "What dimensions are used to rate critiques in the conceptual arguments dataset?",
        "How many rated critiques are included in the dataset?",
        "Does reasoning or thinking generally improve model performance on the conceptual argument dataset?",
        "What two main scoring approaches are used to evaluate models on the dataset?",

        # Paper 2: AI governance and general-purpose AI
        "What four core AI safety properties does the paper discuss?",
        "Why is accuracy easier to evaluate for narrow AI than for general-purpose AI?",
        "Why does the paper argue that explanations produced by general-purpose AI can be misleading?",
        "Why is human-in-the-loop oversight problematic for general-purpose AI performing analytical work?",
        "What major governance changes does the paper recommend for the use of general-purpose AI in policing?",

        # Paper 3: Epistemic Norms for AI Safety and Alignment Research
        "What two asymmetries distinguish AI alignment research from mainstream AI research?",
        "What five epistemic gaps does the paper identify in current AI alignment research?",
        "What is the primary governance target of ECAISA?",
        "What did the bibliometric pilot find about independent verification in alignment research?",
        "Why does the paper use the aviation catastrophic failure rate of 10^-9 only as an analogy rather than a target for AI alignment?"
    ]

    ground_truths = [
        # Paper 1
        "Conceptual questions lack a realistically accessible ground-truth answer or widely accepted method for resolving them, but progress can be made by considering and debating arguments.",

        "Critiques are rated on centrality, strength, correctness, clarity, dead weight, single issue, and an overall holistic rating.",

        "The dataset contains 951 rated critiques.",

        "Reasoning or thinking generally does not improve performance much on the dataset; it sometimes helps and sometimes hurts, with differences usually being small.",

        "The paper uses a weighted pairwise ranking error rate and a custom weighted loss to evaluate model ratings.",

        # Paper 2
        "The four core AI safety properties are accuracy, bias, explainability, and accountability.",

        "Narrow AI performs a predefined task in a bounded input-output space with labelled examples and measurable correctness, so error rates and failure cases can be calculated.",

        "General-purpose AI can generate fluent and convincing explanations that may be post-hoc rationalizations rather than faithful representations of the processes that actually produced the output.",

        "Meaningfully checking a general-purpose AI analysis may require the human reviewer to independently perform the same analytical work. If the human can already do that, the AI may be redundant, while if they cannot, meaningful verification becomes difficult.",

        "The paper recommends distinguishing narrow AI from general-purpose AI in governance, preferring narrow tools when viable, pausing operational GPAI deployment in policing until sufficient evidence exists, and creating independent centralized safety infrastructure.",

        # Paper 3
        "The two asymmetries are capability profile and risk profile: mainstream AI demonstrates positive capabilities and optimizes average-case performance, while alignment must demonstrate the absence of hazardous behavior and address worst-case outcomes under fat-tailed uncertainty.",

        "The five gaps concern the optimization target, visibility or transparency of evidence, trustworthiness through verification and replication, treatment of uncertainty and unknowns, and enforcement culture or organized skepticism.",

        "ECAISA targets auditability rather than certification; it aims to make safety-relevant research claims independently inspectable, contestable, and verifiable rather than certify that an AI system is safe.",

        "The pilot found independent verification in zero percent of the ten alignment papers examined, with none citing an independent replication, third-party audit, or external red-team study of the central claim.",

        "The 10^-9 aviation failure rate is only an analogy because AI alignment lacks actuarial base-rate data, well-defined failure modes, and stationary system dynamics needed to justify such a numerical threshold; the intended lesson is evidence proportionality."
    ]

    rag.evaluate(queries, ground_truths)
