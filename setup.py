from setuptools import setup, find_packages

setup(
    name="financial-qa-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "faiss-cpu",
        "transformers",
        "torch",
        "openai",
        "python-dotenv",
        "pypdf",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "pytest",
        "pyyaml",
        "fastapi",
        "uvicorn",
        "streamlit",
        "sentence-transformers",
        "datasets",
    ],
    author="Pulishetty Anjaneyaprasad",
    author_email="prasadpatel16@gmail.com",
    description="A financial domain-specific question-answering system with hallucination mitigation",
)
