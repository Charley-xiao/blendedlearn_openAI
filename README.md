# blendedlearn_openAI

Generative AI with LLMs and Reinforcement Learning - OpenAI Project
https://touch.larksuite.com/docx/NoxCdt8Q5o6Qh0xDBc1uqlThsfd

Meeting Notes: Open AI Subject Setup Session Part 2 Nov 08, 2024
https://touch.larksuite.com/docx/E4ondyUXxoN2yDxQT4AuVN1Fsuc

## Project Description

Dive into the refinement of the FLAN-T5 model, utilizing reinforcement learning to generate safer, non-toxic content.
- Integrate Meta AI's hate speech reward model, a binary classifier that assesses text as “hate” or “not hate,” guiding the model toward safer language generation.
- Employ Proximal Policy Optimization (PPO) to fine-tune the model by progressively rewarding non-toxic outputs, reducing harmful behavior.
- Gain hands-on experience in advanced AI techniques, such as generative AI, reinforcement learning, and prompt engineering, while applying LLMs to real-world challenges.
- Equip participants with the skills needed to build robust AI systems that mitigate biases and toxic content, enhancing applications in natural language processing, dialogue systems, and content filtering.

## Approach (Subject to Change)

### Training

For a given input text, the PPO trainer generates a response using the current policy of the language model. Each generated response is evaluated using a multi-dimensional reward model. This model incorporates Meta AI's hate speech classifier and additional classifiers for other toxicity aspects like harassment, discrimination, and profanity. The reward is computed based on the severity and type of toxicity detected, with non-toxic responses receiving higher rewards. The computed rewards are used to update the language model's policy through PPO.

### Curriculum Learning

Additionally, the training incorporates a curriculum learning strategy where the model is progressively exposed to more challenging prompts. It starts with clear-cut, non-toxic examples and gradually includes ambiguous or potentially harmful inputs. This helps the model adapt to a wide range of scenarios and improves its robustness.

### Constitutional Constraints

A set of predefined ethical guidelines, or a "constitution," is embedded into the training process. The model is trained to adhere to these guidelines by self-evaluating its outputs and making necessary adjustments to comply with ethical standards. This promotes consistency and transparency in the model's behavior.