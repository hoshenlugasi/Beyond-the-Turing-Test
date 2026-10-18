from llm_players.llm_player import LLMPlayer, ABC


class PromptingPlayer(LLMPlayer, ABC):
    pass

    # def turn_task_into_prompt(self, task, message_history):
    #     prompt = f"The current time is [{get_current_timestamp()}].\n"
    #     if not message_history:
    #         prompt += "No player has sent a message yet.\n"
    #     else:
    #         prompt = "Here is the message history so far, including [timestamps]:\n"
    #         prompt += "".join(message_history)  # each one already ends with "\n"
    #     prompt += task.strip() + " "  # validate a " " is added before the next instruction
    #     # not necessarily needed with all models, seemed relevant to Llama3.1:
    #     prompt += "Don't add the time, the timestamp or the [timestamp] in your answer!\n"
    #     return prompt

