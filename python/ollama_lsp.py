from pygls.server import LanguageServer
from lsprotocol import types
from completion_engine import CompletionEngine
import re
import asyncio
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ollama_copilot.log'
)
logger = logging.getLogger('ollama_copilot')

class OllamaServer:
    def __init__(self):
        self.server = LanguageServer("ollama-server", "v0.3")
        self.engine = None
        self.curr_suggestion = {'line': 0, 'character': 0, 'suggestion': ''}
        self.cancel_suggestion = False
        self.debounce_time = 0.5
        self.last_completion_request = None
        self.debounce_task = None
        self.register_features()
    
    def register_features(self):
        @self.server.feature(types.INITIALIZE)
        def initialize(params: types.InitializeParams):
            return self.on_initialize(params)
        
        @self.server.feature(types.TEXT_DOCUMENT_COMPLETION)
        def completions(params: types.CompletionParams):
            return []

        @self.server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
        def change(params: types.DidChangeTextDocumentParams):
            return self.on_change(params)

    def on_initialize(self, params: types.InitializeParams):
        try:
            logger.info("Initializing Ollama LSP server")
            init_options = params.initialization_options
            
            self.engine = CompletionEngine(
                init_options.get('model_name', "deepseek-coder:base"), 
                options=init_options.get('ollama_model_opts', {})
            )
            self.stream_suggestion = init_options.get('stream_suggestion', False)
            
            logger.info(f"Initialized with model: {init_options.get('model_name', 'deepseek-coder:base')}")
            
            return {
                "capabilities": {
                    "textDocumentSync": types.TextDocumentSyncKind.Incremental,
                    "completionProvider": {
                        "resolveProvider": True,
                        "triggerCharacters": [' ', '.', '(', '{', '[', '\n']
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise

    def debounce_completion(self, params: types.CompletionParams):
        self.last_completion_request = params
        if self.debounce_task:
            self.debounce_task.cancel()
        self.debounce_task = asyncio.create_task(self.handle_debounce())
        return []

    async def handle_debounce(self):
        await asyncio.sleep(self.debounce_time)
        if self.last_completion_request:
            params = self.last_completion_request
            self.last_completion_request = None
            await self.on_completion(params)

    async def on_completion(self, params: types.CompletionParams):
        try:
            document = self.server.workspace.get_text_document(params.text_document.uri)
            file_path = document.uri.replace("file://", "")
            lines = document.lines

            logger.info(f"Starting completion for file: {file_path}")
            logger.debug(f"Position - Line: {params.position.line}, Character: {params.position.character}")

            suggestion_stream = self.engine.complete(
                lines, 
                params.position.line, 
                params.position.character,
                file_path=file_path
            )

            self.curr_suggestion = {
                'line': params.position.line + 1, 
                'character': params.position.character, 
                'suggestion': ''
            }
            timing_str = ''

            async for chunk in suggestion_stream:
                if self.cancel_suggestion:
                    logger.info("Completion cancelled")
                    self.cancel_suggestion = False
                    return []

                if 'message' in chunk:
                    self.curr_suggestion['suggestion'] += chunk['message']['content']
                elif 'response' in chunk:
                    self.curr_suggestion['suggestion'] += chunk['response']

                if 'context' in chunk:
                    total_duration = chunk['total_duration'] / 10**9
                    load_duration = chunk['load_duration'] / 10**9
                    prompt_eval_duration = chunk['prompt_eval_duration'] / 10**9
                    eval_count = chunk['eval_count']
                    eval_duration = chunk['eval_duration'] / 10**9
                    timing_str = f"""
                        Total duration: {total_duration},
                        Load duration: {load_duration},
                        Prompt eval duration: {prompt_eval_duration},
                        Eval count: {eval_count},
                        Eval duration: {eval_duration}"""
                    logger.debug(f"Completion timing: {timing_str}")

                if self.stream_suggestion:
                    self.send_suggestion(
                        self.curr_suggestion['suggestion'],
                        self.curr_suggestion['line'],
                        self.curr_suggestion['character'],
                        suggestion_type='stream'
                    )
            
            final_suggestion = self.strip_suggestion(self.curr_suggestion['suggestion'])
            self.send_suggestion(
                final_suggestion,
                self.curr_suggestion['line'],
                self.curr_suggestion['character'],
                suggestion_type='completion'
            )
            
            logger.info(f"Completion finished successfully. Length: {len(final_suggestion)}")
            
        except Exception as e:
            logger.error(f"Error during completion: {str(e)}", exc_info=True)
            self.send_suggestion(
                "",
                self.curr_suggestion['line'],
                self.curr_suggestion['character'],
                suggestion_type='error'
            )
            return []

        return []

    def on_change(self, params: types.DidChangeTextDocumentParams):
        try:
            change = params.content_changes[0]
            logger.debug(f"Document change detected: {change.text}")

            lines = self.server.workspace.get_text_document(params.text_document.uri).lines
            line = lines[change.range.start.line][change.range.start.character + 1:]
            contains_non_whitespace = bool(re.search(r'[^\s]', line))

            if contains_non_whitespace:
                return

            if (change.text == self.curr_suggestion['suggestion'][0:len(change.text)] and 
                len(change.text) > 0):
                self.curr_suggestion['suggestion'] = self.curr_suggestion['suggestion'][len(change.text):]
                self.curr_suggestion['character'] += len(change.text)
                self.send_suggestion(
                    self.curr_suggestion['suggestion'],
                    self.curr_suggestion['line'],
                    self.curr_suggestion['character'],
                    suggestion_type='fill_suggestion'
                )
                return
            else:
                self.curr_suggestion = {'line': 1, 'character': 0, 'suggestion': ''}
                self.clear_suggestion()

                position = types.Position(
                    line=change.range.end.line, 
                    character=change.range.end.character + 1
                )
                completion_params = types.CompletionParams(
                    text_document=params.text_document,
                    position=position,
                    context=types.CompletionContext(
                        trigger_kind=types.CompletionTriggerKind.Invoked
                    )
                )

                if len(change.text) == 0:
                    return
                self.debounce_completion(completion_params)

        except Exception as e:
            logger.error(f"Error handling change event: {str(e)}", exc_info=True)

    def clear_suggestion(self):
        self.server.send_notification('$/clearSuggestion', {'message': "clear current"})

    def send_suggestion(self, suggestion, line, col, suggestion_type='miscellaneous'):
        self.server.send_notification('$/tokenStream', {
            'line': line,
            'character': col,
            'completion': {
                'total': suggestion,
                'type': suggestion_type,
            }
        })

    def strip_suggestion(self, text):
        stripped_text = text.rstrip('\n')
        return re.sub(r'\n{2,}', '\n', stripped_text)
    
    def start(self):
        try:
            logger.info("Starting Ollama LSP server")
            self.server.start_io()
        except Exception as e:
            logger.error(f"Error starting server: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    try:
        server = OllamaServer()
        server.start()
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}", exc_info=True)
        raise
