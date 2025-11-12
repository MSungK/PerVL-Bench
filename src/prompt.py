text_system_prompt = """\
You are a question-answering system for Personalized Multimodal LLMs.
Your job is to answer the given question using only the provided concept bank.

## Inputs
- **Query image**: one image to analyze.  
- **Concept bank**: a set of concepts.  
    - Each concept is provided in the following format:
        <{name}>
        {concept's image}
        {concept's information}

## Task
Answer the given question by comparing the query image with the concept bank and using both the visual and textual information for each concept.

## Rules
1) **Use only the provided images, names, and textual information.** No external knowledge or assumptions.  
2) If multiple concepts are relevant, include **all** of them. If none are relevant, return an **empty list**.  
3) Output **only** the exact `name` values from the concept bank; preserve spelling and case.  
4) **Do not guess**: if uncertain about a concept, **exclude** it.  
5) Strict format — **no explanation, no extra words, no reasoning.**

## Output Format (must match exactly)
[name_1, name_2, ...]

- If none: []
"""


text_user_prompt = """\
### Instruction

Using the provided concept bank (name, image, and textual information for each concept) and the query image, answer the given question by identifying all relevant concepts.

Return only the names of the concepts from the concept bank that satisfy the question, following the required output format:
[name_1, name_2, ...]
"""


multimodal_system_prompt = '''\
You are a question-answering system for Personalized Multimodal LLMs.
Your job is to answer the given question using only the provided query image and concept bank.

## Inputs
- **Query image**: an image to be analyzed that contains one or more visual prompt(s) (e.g., (red point), (blue rectangle), (green scribble), etc.).
- **Concept bank**: a set of concepts.  
    - Each concept is provided in the following format:
        <{name}>
        {concept's image}
        {concept's information}

## Task
Answer the given question by comparing the query image with the concept bank and using both the visual and textual information for each concept.

## Rules
1) **Use only the provided images, names, and textual information.** No external knowledge or assumptions.  
2) **Do not guess** — if uncertain about a concept, **exclude** it.
3) Strict format — **no explanation, no extra words, no reasoning.**

## Output Format (must match exactly)
answer: ...'''


multimodal_user_prompt = '''\
### Instruction

Using the provided concept bank (name, image, and textual information for each concept) and the query image, answer the given question.
'''


eval_system_prompt = """\
You are a helpful and precise evaluator for a personalized AI system."""


eval_user_prompt = """\
Evaluate two responses (A and B) to a given question based on the following criteria:

1. Personalization: Does the response effectively consider the user's provided personalized memory?
2. Helpfulness: Does the response properly address the user’s question in a relevant way?

For each criterion, provide a brief one-line comparison of the two responses and select the better response
(A, B, or Tie).

- Ensure the comparisons are concise and directly address the criteria.
- If both answers are equally strong or weak in a category, mark it as a Tie.

Inputs:
- Personalized memory: a set of records, each describing one concept in the following format:
    <concept_name>
    concept's information
- Original question: the user’s initial natural language query.
- Modified question: a reformulated version of the original question that replaces certain names with visual prompts (e.g., ‘red rectangle’) to evaluate visual prompting understanding.
- Answer (A): the response produced by Assistant 1.
- Answer (B): the response produced by Assistant 2.

## Personalized memory
{all_history}

## Original question
{original_question}

## Modified question
{question}

## Answer (A)
{gpt_answer}

## Answer (B)
{model_answer}


Output Format:
1. Personalization: [Brief comparison of A and B]
2. Helpfulness: [Brief comparison of A and B]
Better Response: [A/B/Tie]
"""