from typing import Dict, List, Any

from tasks.extraction import extract_ground_truth_questions
from registry import TaskRegistry
from .base import Task
import sacrebleu
import re
from dataclasses import dataclass


@TaskRegistry.register("socratic_questioning")
class SocraticQuestioningTask(Task):
    def parse_response(self, response: str) -> List[str]:
        """Extract questions from the model's response"""
        # The model may include the entire conversation history, so we need to extract
        # only the FIRST set of questions (right after the prompt's "Questions:" marker)
        # and BEFORE any "Human:" or "Problem:" markers (which indicate conversation history)
        questions = []
        
        # Strategy: 
        # 1. If there's a "Human:" or "Problem:" marker, take everything BEFORE it
        #    (this is the actual response, before conversation history starts)
        # 2. Then extract questions from that portion
        
        # First, cut off at conversation history markers
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        if "Problem:" in response:
            response = response.split("Problem:")[0].strip()
        
        # Now handle "Questions:" marker - if present, take what comes after the FIRST occurrence
        # (the prompt ends with "Questions:", so the model's response should follow)
        if "Questions:" in response:
            parts = response.split("Questions:", 1)  # Split only on first occurrence
            if len(parts) > 1:
                response = parts[1].strip()  # Take what comes after "Questions:"
        
        # Also handle "Assistant:" marker - take what comes after it if present
        if "Assistant:" in response:
            parts = response.split("Assistant:")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        # Remove any remaining "Human:" or "Problem:" lines (these are from conversation history)
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that are clearly conversation history markers
            if line and not line.startswith(('Human:', 'Problem:', 'Assistant:')):
                cleaned_lines.append(line)
        
        # Rejoin and process
        response = ' '.join(cleaned_lines)
        
        # Split on '?' since questions are on one line separated by question marks
        for question in response.split('?'):
            question = question.strip()
            # Filter out common prefixes and clean up
            if question:
                # Remove any remaining prefixes
                question = re.sub(r'^(Human:|Problem:|Assistant:)\s*', '', question)
                # Remove any newlines or extra whitespace
                question = re.sub(r'\s+', ' ', question).strip()
                # Only add if it looks like a real question (has some content, not just punctuation)
                # and doesn't contain conversation markers
                if (question and len(question) > 3 and 
                    not question.startswith(('?', '!', '.')) and
                    'Human:' not in question and 
                    'Problem:' not in question and
                    'Assistant:' not in question and
                    'Generate only' not in question):
                    questions.append(question + '?')  # Add back the '?' that was removed by split
        
        # If we got too many questions (likely parsing error), try to take only the first few
        # that look legitimate (not containing conversation markers)
        if len(questions) > 10:
            filtered_questions = []
            for q in questions:
                # Skip questions that contain conversation history markers
                if not any(marker in q for marker in ['Human:', 'Problem:', 'Assistant:', 'Generate only']):
                    filtered_questions.append(q)
                if len(filtered_questions) >= 10:  # Reasonable max
                    break
            if filtered_questions:
                questions = filtered_questions
        
        return questions


    def compute_metrics(self, predictions: List[List[str]], targets: List[str]) -> Dict[str, float]:
        """Compute SACREBLEU scores for generated questions"""
        # Extract ground truth questions for each target
        print(targets)
        print(predictions)

        target_questions = [extract_ground_truth_questions(target) for target in targets]

        print(target_questions)
        print(predictions)

        # Compute SACREBLEU scores for each example
        scores = []
        question_counts = []


        for pred_questions, target_questions_list in zip(predictions, target_questions):
            if not pred_questions or not target_questions_list:
                continue

            # Create references list in format required by SACREBLEU
            references = [q for q in target_questions_list]

            # Calculate SACREBLEU for each predicted question against all reference questions
            question_scores = []
            for pred in pred_questions:
                bleu = sacrebleu.sentence_bleu(pred, references)
                question_scores.append(bleu.score)

            if question_scores:
                scores.append(max(question_scores))  # Use best match for each prediction
                question_counts.append(min(len(pred_questions), len(target_questions_list)))

        avg_bleu = sum(scores) / len(scores) if scores else 0.0
        avg_questions = sum(question_counts) / len(question_counts) if question_counts else 0.0

        return {
            "bleu": avg_bleu / 100.0,  # Normalize to 0-1 range
            "avg_questions": avg_questions
        }
