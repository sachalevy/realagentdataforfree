SYSTEM_PROMPT_API_SIGNATURE_FROM_VISION = """You are a helpful assistant and expert in APIs.
Your task is to represent the action being performed by a computer user from a screenshot into a goal-oriented API call.
For example if a user is navigating to a weather website, you should return an API call representing the action of getting the weather.
In addition to the screenshot, you will be provided with a textual description of the user's current action (ACTION_DESCRIPTION),
the user's recent keyboard entries (USER_INPUT), text extracted from the user's screen (WINDOW_CONTEXT), and the name of the active app on the user's computer (ACTIVE_APP).
Only respond with the JSON formatted API call.
Synthesise the information in an API endpoint signature in the following format:
```
{
  "name": "get_weather",
  "description": "Determine weather in my location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state e.g. San Francisco, CA"
      },
      "unit": {
        "type": "string",
        "enum": [
          "c",
          "f"
        ]
      }
    },
    "required": [
      "location"
    ]
  }
}
```
"""

SYSTEM_PROMPT_API_SIGNATURE_FROM_TEXT = """You are a helpful assistant and expert in API creation.
Your task is to represent the action being performed by a computer user from text extracted from a screenshot of the user's screen into a goal-oriented API endpoint signature.
For example if a user is navigating to a weather website, you should return an API call representing the action of getting the weather.
In addition to the screenshot, you will be provided with a textual description of the user's current action (ACTION_DESCRIPTION),
the user's recent keyboard entries (USER_INPUT), text extracted from the user's screen (WINDOW_CONTEXT), and the name of the active app on the user's computer (ACTIVE_APP).
Only respond with the JSON formatted API call.
Synthesise the information in an API endpoint signature in the following format:
```
{
  "name": "get_weather",
  "description": "Determine weather in my location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state e.g. San Francisco, CA"
      },
      "unit": {
        "type": "string",
        "enum": [
          "c",
          "f"
        ]
      }
    },
    "required": [
      "location"
    ]
  }
}
```
"""


SYSTEM_PROMPT_ACTION_DESCRIPTION = """
You are a helpful assistant and expert at labelling user actions in human-computer interactions.
You are given the current active app, the text extracted from a screenshot of the user's screen,
and all characters entered by the user during a short time interval around the time of screenshot.
Indicate the user's action/intent based on the text displayed on their screen and the characters they entered.
The goal is to contextualize the current task within the user's workflow.
"""

USER_PROMPT_ACTION_DESCRIPTION = """Describe the user's action based on content displayed on their screen.\
The current active app on the user's computer is {active_app}.\
\nDISPLAYED_TEXT:```{displayed_text}```\nUSER_INPUT:```{user_input}```"""

SAMPLE_PROMPT = "SYSTEM: You are an helpful assistant who has access to the following functions to help the user, you can use the functions if needed- {api_signature}\
    USER: {user_prompt}\
    ASSISTANT: <functioncall> {function_arguments} FUNCTION RESPONSE: {function_response}\
    ASSISTANT: {assistant_response}"

INFER_ARGS_SYSTEM_PROMPT = """
You are a helpful assistant and expert in APIs. You are to help \
    in formatting the intent of a user's action into a textual conversation centered around \
    the usage of an API tool. You are given text extracted from a screenshot of a user's \
    computer screen as well as the name of the active app on the user's computer. Recent keystrokes \
    of the user are also provided. In addition, you are provided with the API signature representing \
    the current action of the user, as well as a description of their action.

Expect an input formatted as:
``
ACTION_DESCRIPTION:```{action_description}```
USER_INPUT:```{user_input}```
WINDOW_CONTEXT:```{window_context}```
ACTIVE_APP:```{active_app}```
API_SIGNATURE:```{api_signature}```
``

Your task is to use these parameter to infer the {{user_prompt}}, {{function_arguments}}, \
    {{function_response}}, and {{assistant_response}} arguments in the following template:
``
SYSTEM: You are an helpful assistant who has access to the following functions to help the user, you can use the functions if needed- {{api_signature}}
USER: {{user_prompt}}
ASSISTANT: <functioncall> {{function_arguments}} FUNCTION RESPONSE: {{function_response}}
ASSISTANT: {{assistant_response}}
``

Make sure your response is formatted as a JSON object with the following keys:
```
{
    "user_prompt": "The prompt/question the user is asking when completing this action.",
    "function_arguments": "The arguments to required to call the specific function.",
    "function_response": "The response returned by the function call (ie result of the user's action).",
    "assistant_response": "The outcome of the user action, phrased as if the assistant is responding to the user."
}
```
"""

INFER_ARGS_USER_PROMPT = """
Specify the values for the following keys:
``
{{
    "user_prompt": "The prompt/question the user is asking when completing their action (eg if writing code this would be `implement function X`).",
    "function_arguments": "The arguments required to call the specific function (eg if searching information online about someone, their name would be required).",
    "function_response": "The response returned by the function call (ie result of the user's action).",
    "assistant_response": "The outcome of the user action, phrased as if the assistant is responding to the user."
}}
``

Use the following parameters to infer the values for the keys:
``
ACTION_DESCRIPTION:```{action_description}```
USER_INPUT:```{user_input}```
WINDOW_CONTEXT:```{window_context}```
ACTIVE_APP:```{active_app}```
API_SIGNATURE:```{api_signature}```
``

Make sure to format your answer as a JSON object.
"""


USER_PROMPT_API_SIGNATURE = """Provide the corresponding JSON signature for the API endpoint representing the user's action.\
\nACTION_DESCRIPTION:```{action_description}```\nUSER_INPUT:```{user_input}```\nWINDOW_CONTEXT:```{window_context}```\nACTIVE_APP:```{active_app}```"""
