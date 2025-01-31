import ollama
from ollama import generate
class CompletionEngine:
    def __init__(self, model: str, options={}):
        self.model = model
        self.client = ollama.Client("http://localhost:11434")
        self.options = options

    def create_system_prompt(self, file_path):
        return {
            "role": "system",
            "content": f"""
            Instructions:
                - You are an AI programming assistant.
                - Given a piece of code with the cursor location marked by <CURSOR>, replace <CURSOR> with the correct code.
                - First, think step-by-step.
                - Describe your plan for what to build in pseudocode, written out in great detail.
                - Then output the code replacing the <CURSOR>.
                - Ensure that your completion fits within the language context of the provided code snippet.
                - Ensure, completion is what ever is needed, dont write beyond 1 or 2 line, unless the <CURSOR> is on start of a function, class or any control statment(if, switch, for, while).

            Rules:
                - Only respond with code.
                - Only replace <CURSOR>; do not include any previously written code.
                - Never include <CURSOR> in your response.
                - Handle ambiguous cases by providing the most contextually appropriate completion.
                - Be consistent with your responses.
                - You should only generate code in the language specified in the META_DATA.
                - Never mix text with code.
                - your code should have appropriate spacing.

            META_DATA: 
            {file_path}"""
        }

    def get_cursor_text(self, lines, line, character):
        if line < 0 or line >= len(lines):
            return "\n".join(lines)
        
        target_line = lines[line]
        if character < 0 or character > len(target_line):
            return "\n".join(lines)
        
        lines[line] = target_line[:character] + "<CURSOR>" + target_line[character:]
        return "\n".join(lines)

    def complete(self, lines, line, character, file_path="unknown"):
        cursor_text = self.get_cursor_text(lines, line, character)
        
        messages = [
            self.create_system_prompt(file_path),
            {"role": "user", "content": cursor_text}
        ]

        return self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options=self.options
        )
    
    def fim_complete(self, lines, line, character):
        # Keep FIM completion as a fallback or alternative method
        lines[line] = lines[line][:character] + "<｜fim▁hole｜>" + lines[line][character:]
        content = '<｜fim▁begin｜>' + "\n".join(lines) + '<｜fim▁end｜>'
        return self.client.generate(
            model=self.model, 
            prompt=content,
            stream=True,
            options=self.options
        )
