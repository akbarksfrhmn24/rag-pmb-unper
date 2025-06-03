# RAG System Evaluation

This directory contains tools for evaluating the Retrieval-Augmented Generation (RAG) system. The evaluation framework focuses on assessing the performance of both retrieval and generation components.

## Setup

Before running the evaluation, install the required dependencies:

```bash
pip install -r evaluation_requirements.txt
```

## Usage

### Creating Test Data

To create a template for test data:

```bash
python evaluate_rag.py --create-template
```

This will generate a `test_data_template.json` file that you can fill in with your questions and expected answers.

### Generating Sample Questions

To automatically generate sample questions from your vector store content:

```bash
python evaluate_rag.py --generate-questions 10 --output-dir ./evaluation_results
```

This will create 10 sample questions based on your existing document collection.

### Running Evaluation

To run the evaluation on your test data:

```bash
python evaluate_rag.py --test-data test_data_sample.json --output-dir ./evaluation_results
```

## Evaluation Metrics

The evaluation framework measures the following metrics:

### Retrieval Metrics

- **Precision**: How many of the retrieved documents are relevant
- **Recall**: How many of the relevant documents were retrieved
- **F1 Score**: Harmonic mean of precision and recall
- **Retrieval Time**: How long it takes to retrieve relevant documents

### Answer Quality Metrics

- **Semantic Similarity**: How semantically similar the generated answer is to the expected answer
- **ROUGE Scores**: Lexical overlap between generated and expected answers
  - ROUGE-1: Overlap of unigrams
  - ROUGE-2: Overlap of bigrams
  - ROUGE-L: Longest common subsequence

### Performance Metrics

- **Retrieval Time**: Time spent on document retrieval
- **Generation Time**: Time spent on answer generation
- **Total Response Time**: Total time to generate an answer

## Output

The evaluation results are saved in the specified output directory:

- Individual JSON files for each question
- An overall JSON summary
- A CSV file with all the metrics

## Sample Test Data

A sample test data file (`test_data_sample.json`) is provided to help you get started with the evaluation.
