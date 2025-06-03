# app/evaluate_rag.py
"""
This module provides tools for evaluating the RAG (Retrieval-Augmented Generation) system.
It includes functions for assessing retrieval quality, answer quality, and overall system performance.
"""

import json
import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from app.query import answer_query
from app.embedding import LocalEmbedding
from app.vector_store import get_vector_store


class RAGEvaluator:
    """Class to evaluate the RAG system using various metrics."""
    
    def __init__(self, 
                 test_data_path: str = None, 
                 output_path: str = './evaluation_results'):
        """
        Initialize the RAG evaluator.
        
        Args:
            test_data_path: Path to test data in JSON format with question-answer pairs
            output_path: Directory to save evaluation results
        """
        self.test_data = self._load_test_data(test_data_path) if test_data_path else []
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.embedding_model = LocalEmbedding()
        self.rouge = Rouge()
        self.results = {
            'retrieval_metrics': [],
            'answer_metrics': [],
            'latency_metrics': [],
            'overall': {}
        }
    
    def _load_test_data(self, path: str) -> List[Dict[str, str]]:
        """Load test questions and expected answers from a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate the test data format
            if not isinstance(data, list):
                raise ValueError("Test data should be a list of question-answer pairs")
            
            for item in data:
                if not isinstance(item, dict) or 'question' not in item:
                    raise ValueError("Each test item should have at least a 'question' field")
                    
            return data
        except Exception as e:
            print(f"Error loading test data: {e}")
            return []
    
    def create_test_data_template(self, output_path: str = 'test_data_template.json'):
        """Create a template for test data that users can fill in."""
        template = [
            {
                "question": "Sample question 1?",
                "expected_answer": "Sample expected answer 1.", 
                "ground_truth_docs": ["Optional: known relevant document names or chunks"]
            },
            {
                "question": "Sample question 2?",
                "expected_answer": "Sample expected answer 2.",
                "ground_truth_docs": ["Optional: known relevant document names or chunks"]
            }
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        print(f"Test data template created at {output_path}")
        return output_path
    
    def evaluate_retrieval(self, question: str, ground_truth_docs: List[str] = None) -> Dict[str, float]:
        """
        Evaluate the retrieval component of the RAG system.
        
        Args:
            question: The query question
            ground_truth_docs: List of relevant document chunks or sources
            
        Returns:
            Dict with retrieval metrics
        """
        # Get the vector store
        vector_store = get_vector_store()
        
        # Get the question embedding
        start_time = time.time()
        question_embedding = self.embedding_model.embed_query(question)
        
        # Get the retrieved documents
        results = vector_store.similarity_search_by_vector(question_embedding, k=4)
        retrieval_time = time.time() - start_time
        
        # Extract retrieved document sources
        retrieved_docs = [doc.metadata.get('source', 'unknown') for doc in results]
        
        metrics = {
            'retrieval_time': retrieval_time,
            'num_retrieved': len(retrieved_docs),
            'retrieved_sources': retrieved_docs
        }
        
        # If ground truth documents are provided, calculate precision/recall metrics
        if ground_truth_docs:
            # Convert to sets for easier comparison
            retrieved_set = set(retrieved_docs)
            ground_truth_set = set(ground_truth_docs)
            
            if len(ground_truth_set) > 0:
                # Calculate precision, recall, and F1 score
                true_positives = len(retrieved_set.intersection(ground_truth_set))
                precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0
                recall = true_positives / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics.update({
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
        
        return metrics
    
    def evaluate_answer(self, 
                       question: str, 
                       generated_answer: str, 
                       expected_answer: str = None) -> Dict[str, float]:
        """
        Evaluate the quality of the generated answer.
        
        Args:
            question: The query question
            generated_answer: The answer generated by the RAG system
            expected_answer: The ground truth answer if available
            
        Returns:
            Dict with answer quality metrics
        """
        metrics = {
            'answer_length': len(generated_answer)
        }
        
        # Only calculate these metrics if an expected answer is provided
        if expected_answer:
            # Calculate semantic similarity between generated and expected answers
            gen_embedding = self.embedding_model.embed(generated_answer)
            exp_embedding = self.embedding_model.embed(expected_answer)
            
            # Convert to numpy arrays for cosine similarity calculation
            gen_embedding_np = np.array(gen_embedding).reshape(1, -1)
            exp_embedding_np = np.array(exp_embedding).reshape(1, -1)
            
            # Calculate cosine similarity
            semantic_similarity = cosine_similarity(gen_embedding_np, exp_embedding_np)[0][0]
            
            # Calculate ROUGE scores (lexical overlap)
            try:
                rouge_scores = self.rouge.get_scores(generated_answer, expected_answer)[0]
                
                metrics.update({
                    'semantic_similarity': float(semantic_similarity),
                    'rouge_1_f': rouge_scores['rouge-1']['f'],
                    'rouge_2_f': rouge_scores['rouge-2']['f'],
                    'rouge_l_f': rouge_scores['rouge-l']['f']
                })
            except Exception as e:
                print(f"Error calculating ROUGE scores: {e}")
                metrics.update({
                    'semantic_similarity': float(semantic_similarity),
                    'rouge_error': str(e)
                })
                
        return metrics
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the evaluation on the test dataset.
        
        Returns:
            Dict containing evaluation results
        """
        if not self.test_data:
            print("No test data available. Please provide test data or create it.")
            return self.results
        
        total_retrieval_time = 0
        total_answer_time = 0
        all_precision = []
        all_recall = []
        all_f1 = []
        all_semantic_similarity = []
        all_rouge_l = []
        
        for i, test_item in enumerate(self.test_data):
            question = test_item['question']
            expected_answer = test_item.get('expected_answer')
            ground_truth_docs = test_item.get('ground_truth_docs')
            
            print(f"\nEvaluating question {i+1}/{len(self.test_data)}: {question}")
            
            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(question, ground_truth_docs)
            total_retrieval_time += retrieval_metrics['retrieval_time']
            
            if 'precision' in retrieval_metrics:
                all_precision.append(retrieval_metrics['precision'])
                all_recall.append(retrieval_metrics['recall'])
                all_f1.append(retrieval_metrics['f1_score'])
            
            # Time the full query process
            start_time = time.time()
            result = answer_query(question)
            answer_time = time.time() - start_time
            total_answer_time += answer_time
            
            generated_answer = result['answer']
            
            # Evaluate answer quality
            answer_metrics = self.evaluate_answer(question, generated_answer, expected_answer)
            
            if 'semantic_similarity' in answer_metrics:
                all_semantic_similarity.append(answer_metrics['semantic_similarity'])
            
            if 'rouge_l_f' in answer_metrics:
                all_rouge_l.append(answer_metrics['rouge_l_f'])
            
            # Record all metrics for this question
            question_results = {
                'question': question,
                'generated_answer': generated_answer,
                'expected_answer': expected_answer,
                'retrieval_metrics': retrieval_metrics,
                'answer_metrics': answer_metrics,
                'total_time': answer_time,
                'retrieval_time': retrieval_metrics['retrieval_time'],
                'generation_time': answer_time - retrieval_metrics['retrieval_time']
            }
            
            self.results['retrieval_metrics'].append(retrieval_metrics)
            self.results['answer_metrics'].append(answer_metrics)
            self.results['latency_metrics'].append({
                'retrieval_time': retrieval_metrics['retrieval_time'],
                'total_time': answer_time,
                'generation_time': answer_time - retrieval_metrics['retrieval_time']
            })
            
            # Save individual result
            with open(self.output_path / f"question_{i+1}_results.json", 'w', encoding='utf-8') as f:
                json.dump(question_results, f, indent=2, ensure_ascii=False)
        
        # Calculate overall metrics
        num_questions = len(self.test_data)
        self.results['overall'] = {
            'num_questions': num_questions,
            'avg_retrieval_time': total_retrieval_time / num_questions if num_questions > 0 else 0,
            'avg_total_time': total_answer_time / num_questions if num_questions > 0 else 0,
            'avg_precision': np.mean(all_precision) if all_precision else None,
            'avg_recall': np.mean(all_recall) if all_recall else None,
            'avg_f1': np.mean(all_f1) if all_f1 else None,
            'avg_semantic_similarity': np.mean(all_semantic_similarity) if all_semantic_similarity else None,
            'avg_rouge_l': np.mean(all_rouge_l) if all_rouge_l else None
        }
        
        # Save overall results
        with open(self.output_path / "overall_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results['overall'], f, indent=2, ensure_ascii=False)
            
        print("\nEvaluation completed!")
        print(f"Results saved to {self.output_path}")
        
        return self.results
    
    def export_results_to_csv(self, file_path: str = None) -> str:
        """Export evaluation results to a CSV file."""
        if file_path is None:
            file_path = str(self.output_path / "evaluation_results.csv")
            
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'Question',
                'Retrieval Time (s)',
                'Generation Time (s)',
                'Total Time (s)',
                'Precision',
                'Recall',
                'F1 Score',
                'Semantic Similarity',
                'ROUGE-L F1'
            ])
            
            # Write data for each question
            for i, test_item in enumerate(self.test_data):
                question = test_item['question']
                retrieval_metrics = self.results['retrieval_metrics'][i]
                answer_metrics = self.results['answer_metrics'][i]
                latency_metrics = self.results['latency_metrics'][i]
                
                writer.writerow([
                    question,
                    f"{retrieval_metrics.get('retrieval_time', 'N/A'):.4f}",
                    f"{latency_metrics.get('generation_time', 'N/A'):.4f}",
                    f"{latency_metrics.get('total_time', 'N/A'):.4f}",
                    f"{retrieval_metrics.get('precision', 'N/A'):.4f}" if 'precision' in retrieval_metrics else 'N/A',
                    f"{retrieval_metrics.get('recall', 'N/A'):.4f}" if 'recall' in retrieval_metrics else 'N/A',
                    f"{retrieval_metrics.get('f1_score', 'N/A'):.4f}" if 'f1_score' in retrieval_metrics else 'N/A',
                    f"{answer_metrics.get('semantic_similarity', 'N/A'):.4f}" if 'semantic_similarity' in answer_metrics else 'N/A',
                    f"{answer_metrics.get('rouge_l_f', 'N/A'):.4f}" if 'rouge_l_f' in answer_metrics else 'N/A'
                ])
                
        print(f"Evaluation results exported to {file_path}")
        return file_path
    
    def generate_sample_questions(self, num_questions: int = 5, output_path: str = None) -> List[Dict]:
        """Generate sample questions from the vector store content for evaluation."""
        # Get the vector store
        vector_store = get_vector_store()
        
        # Get all documents in the vector store
        docs = vector_store.similarity_search("", k=50)  # Get a sample of documents
        
        if not docs:
            print("No documents found in the vector store.")
            return []
        
        # Create questions based on document content
        sample_questions = []
        processed_content = set()  # To avoid duplicate content
        
        for doc in docs:
            # Skip if we already have enough questions
            if len(sample_questions) >= num_questions:
                break
                
            content = doc.page_content.strip()
            source = doc.metadata.get('source', 'unknown')
            
            # Skip if we've already processed similar content
            content_hash = hash(content[:100])  # Use first 100 chars as a proxy for content
            if content_hash in processed_content:
                continue
                
            processed_content.add(content_hash)
            
            # Create a question
            sample_question = {
                "question": f"Berikan informasi tentang {content.split()[:5]}?",  # Simple question about first few words
                "expected_answer": "",  # To be filled by the user
                "ground_truth_docs": [source]
            }
            
            sample_questions.append(sample_question)
        
        # Save to output file if specified
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sample_questions, f, indent=2, ensure_ascii=False)
            print(f"Sample questions saved to {output_path}")
        
        return sample_questions


def main():
    """Run the RAG evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate the RAG system')
    parser.add_argument('--test-data', type=str, help='Path to test data JSON file')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', 
                        help='Directory to save evaluation results')
    parser.add_argument('--create-template', action='store_true', 
                        help='Create a template for test data')
    parser.add_argument('--generate-questions', type=int, 
                        help='Generate sample questions from vector store content')
    
    args = parser.parse_args()
    
    evaluator = RAGEvaluator(args.test_data, args.output_dir)
    
    if args.create_template:
        evaluator.create_test_data_template()
        return
        
    if args.generate_questions:
        samples = evaluator.generate_sample_questions(
            args.generate_questions, 
            output_path=f"{args.output_dir}/sample_questions.json"
        )
        print(f"Generated {len(samples)} sample questions")
        return
    
    if not args.test_data:
        print("No test data provided. Creating a template...")
        evaluator.create_test_data_template()
        print("Please fill in the template with your test questions and run again.")
        return
    
    results = evaluator.run_evaluation()
    evaluator.export_results_to_csv()
    
    # Print summary
    overall = results['overall']
    print("\n=== RAG Evaluation Summary ===")
    print(f"Number of questions: {overall['num_questions']}")
    print(f"Average retrieval time: {overall['avg_retrieval_time']:.4f} seconds")
    print(f"Average total response time: {overall['avg_total_time']:.4f} seconds")
    
    if overall['avg_precision'] is not None:
        print(f"Average precision: {overall['avg_precision']:.4f}")
        print(f"Average recall: {overall['avg_recall']:.4f}")
        print(f"Average F1 score: {overall['avg_f1']:.4f}")
    
    if overall['avg_semantic_similarity'] is not None:
        print(f"Average semantic similarity: {overall['avg_semantic_similarity']:.4f}")
    
    if overall['avg_rouge_l'] is not None:
        print(f"Average ROUGE-L F1: {overall['avg_rouge_l']:.4f}")


if __name__ == "__main__":
    main()
