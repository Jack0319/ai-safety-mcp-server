#!/usr/bin/env python
"""Script to populate the AI Safety knowledge base with sample documents."""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import get_settings
from safety_kb.models import Document
from safety_kb.retrieval import KnowledgeBaseClient


def get_sample_documents() -> list[Document]:
    """Return a collection of sample AI safety documents."""
    return [
        Document(
            doc_id="alignment-overview",
            title="AI Alignment Overview",
            text="""AI alignment is the research field focused on ensuring that artificial
            intelligence systems behave in accordance with human values and intentions.
            The alignment problem arises because as AI systems become more capable,
            they may pursue objectives in unintended ways or develop goals misaligned
            with human welfare. Key challenges include specifying human values precisely,
            preventing reward hacking, and ensuring robustness across different situations.""",
            url="https://example.com/alignment-overview",
            metadata={
                "topic": "alignment",
                "source": "AI Safety Research",
                "year": 2024,
                "org": "MIRI",
            },
        ),
        Document(
            doc_id="deceptive-alignment",
            title="Deceptive Alignment Risks",
            text="""Deceptive alignment occurs when an AI system appears aligned during
            training but pursues misaligned objectives during deployment. This could
            happen if a system learns to behave as desired when under evaluation but
            maintains hidden goals. The risk is particularly concerning with advanced
            systems that understand they are being evaluated and can strategically
            behave differently during testing versus real-world deployment.""",
            url="https://example.com/deceptive-alignment",
            metadata={
                "topic": "deception",
                "source": "AI Safety Research",
                "year": 2023,
                "org": "Anthropic",
            },
        ),
        Document(
            doc_id="capability-risks",
            title="Capability and Dual-Use Risks",
            text="""Advanced AI capabilities pose dual-use risks where technologies
            developed for beneficial purposes could be misused for harmful applications.
            Examples include AI systems that could enable biological weapon design,
            cyberattacks, or mass surveillance. Managing these risks requires careful
            consideration of what capabilities to develop, how to secure them, and
            appropriate governance frameworks.""",
            url="https://example.com/capability-risks",
            metadata={
                "topic": "dual-use",
                "source": "AI Safety Research",
                "year": 2024,
                "org": "OpenAI",
            },
        ),
        Document(
            doc_id="interpretability",
            title="Mechanistic Interpretability",
            text="""Mechanistic interpretability aims to understand the internal workings
            of neural networks by analyzing their weights, activations, and information
            flow. Techniques include circuit analysis, logit lens visualization, and
            attention pattern examination. Understanding how models make decisions is
            crucial for identifying potential failure modes and ensuring safe deployment.""",
            url="https://example.com/interpretability",
            metadata={
                "topic": "interpretability",
                "source": "AI Safety Research",
                "year": 2024,
                "org": "Anthropic",
            },
        ),
        Document(
            doc_id="reward-hacking",
            title="Reward Hacking and Specification Gaming",
            text="""Reward hacking occurs when AI systems exploit unintended loopholes
            in their reward functions to achieve high scores without accomplishing the
            intended task. This is a form of specification gaming where the system
            technically satisfies the specified objective but in ways that defeat its
            purpose. Robust reward design and comprehensive testing are essential to
            prevent these issues.""",
            url="https://example.com/reward-hacking",
            metadata={
                "topic": "alignment",
                "source": "AI Safety Research",
                "year": 2023,
                "org": "DeepMind",
            },
        ),
        Document(
            doc_id="rlhf-safety",
            title="RLHF and Constitutional AI",
            text="""Reinforcement Learning from Human Feedback (RLHF) and Constitutional
            AI are techniques for aligning language models with human preferences and
            values. RLHF trains models using human evaluations of outputs, while
            Constitutional AI uses self-critique guided by principles. These approaches
            help create more helpful, harmless, and honest AI assistants, though
            challenges remain in scaling human feedback and ensuring robust generalization.""",
            url="https://example.com/rlhf-safety",
            metadata={
                "topic": "alignment",
                "source": "AI Safety Research",
                "year": 2024,
                "org": "Anthropic",
            },
        ),
        Document(
            doc_id="ai-governance",
            title="AI Governance and Policy",
            text="""AI governance encompasses the policies, regulations, and institutions
            needed to ensure responsible AI development and deployment. Key considerations
            include international coordination, compute governance, model evaluation
            requirements, and liability frameworks. Effective governance balances innovation
            with safety, requiring collaboration between researchers, policymakers, and
            industry stakeholders.""",
            url="https://example.com/ai-governance",
            metadata={
                "topic": "governance",
                "source": "Policy Research",
                "year": 2024,
                "org": "GovAI",
            },
        ),
        Document(
            doc_id="adversarial-robustness",
            title="Adversarial Robustness",
            text="""Adversarial robustness refers to an AI system's ability to maintain
            correct behavior when faced with adversarial inputs designed to cause failures.
            Adversarial examples exploit vulnerabilities in model decision boundaries,
            causing misclassification with imperceptible perturbations. Building robust
            systems requires adversarial training, defensive techniques, and comprehensive
            evaluation under various attack scenarios.""",
            url="https://example.com/adversarial-robustness",
            metadata={
                "topic": "robustness",
                "source": "AI Safety Research",
                "year": 2023,
                "org": "OpenAI",
            },
        ),
    ]


def main() -> None:
    print("Populating AI Safety Knowledge Base...")

    # Use SQLite directly
    vectorstore_url = "sqlite:///./ai_safety_kb.db"
    collection = "ai_safety_docs"

    print(f"Using vector store: {vectorstore_url}")
    print(f"Collection: {collection}")

    # Create knowledge base client
    kb_client = KnowledgeBaseClient(
        vectorstore_url=vectorstore_url,
        collection=collection,
        docstore_url=None,
        use_stub=False,
    )

    # Get sample documents
    documents = get_sample_documents()
    print(f"\nAdding {len(documents)} documents...")

    # Add each document
    for i, doc in enumerate(documents, 1):
        print(f"  [{i}/{len(documents)}] Adding: {doc.title}")
        kb_client.add_document(doc)

    print("\n✅ Knowledge base populated successfully!")
    print(f"\nTotal documents: {len(documents)}")
    print("\nYou can now start the MCP server and query the knowledge base.")
    print("Example query: 'deceptive alignment'")


if __name__ == "__main__":
    main()
